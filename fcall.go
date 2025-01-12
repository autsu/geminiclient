package geminiclient

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/google/generative-ai-go/genai"
)

type FuncMetadataManager struct {
	Fms map[string]*FuncMetadata
}

func (fmm *FuncMetadataManager) AddFunction(fm *FuncMetadata) {
	if fm == nil {
		panic("BUG: func metadata is nil")
	}
	fm.init()
	fmm.Fms[fm.FnName] = fm
}

var ErrEmptyPrompt = errors.New("empty prompt")

type Property struct {
	Description string
	Type        genai.Type
	Required    bool
}

type FuncMetadata struct {
	Fn       any                 // 函数
	FnName   string              // 函数名
	FnDesc   string              // 函数描述
	Args     map[string]Property // 函数参数描述
	RetNames []string            // 函数返回值名称

	fnType  reflect.Type
	fnValue reflect.Value
}

func (fm *FuncMetadata) init() {
	if fm.Fn == nil {
		panic("BUG: fm.Fn is nil")
	}
	fm.fnType = reflect.TypeOf(fm.Fn)
	if fm.fnType.Kind() != reflect.Func {
		panic("BUG: Fn is not a function")
	}
	fm.fnValue = reflect.ValueOf(fm.Fn)
}

// 函数参数描述
// key: 参数名
// value:
// type FuncMetadata map[string]Property
func (f *FuncMetadata) FuncMetadataToSchema() *genai.Schema {
	ret := &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{}}
	for k, v := range f.Args {
		ret.Properties[k] = &genai.Schema{
			Type:        v.Type,
			Description: v.Description,
		}
		if v.Required {
			ret.Required = append(ret.Required, k)
		}
	}
	return ret
}

type FunctionCall func(req map[string][]Property) (resp map[string]any, err error)

// FunctionCallHandler defines a callback type for handling function responses.
type FunctionCallHandler func(response map[string]any) (map[string]any, error)

func (gc *GeminiClient) AddFunctionToolNew(fm *FuncMetadata) error {
	if fm == nil {
		panic("BUG: func metadata is nil")
	}
	fm.init()
	gc.FuncMetadataManager.AddFunction(fm)

	schema := fm.FuncMetadataToSchema()
	functionDecl := &genai.FunctionDeclaration{
		Name:        fm.FnName,
		Description: fm.FnDesc,
		Parameters:  schema,
	}

	tool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{functionDecl},
	}
	gc.Tools = append(gc.Tools, tool)

	return nil
}

// AddFunctionTool registers a custom Go function as a tool that the model can call.
func (gc *GeminiClient) AddFunctionTool(name, description string, fn any) error {
	fnValue := reflect.ValueOf(fn)
	fnType := fnValue.Type()

	if fnType.Kind() != reflect.Func {
		return fmt.Errorf("provided argument is not a function")
	}

	parameters := make(map[string]*genai.Schema)
	var required []string

	for i := 0; i < fnType.NumIn(); i++ {
		paramType := fnType.In(i)
		paramName := fmt.Sprintf("param%d", i+1)

		// 貌似无法通过反射拿到参数名，所以这里塞的就是 param1, param2 这样的东西
		// 也无法提供 Description 信息
		// 这样似乎会影响 genmini 调用时提供参数的准确性
		// 比如我有一个获取气温的函数，两个参数分别是 location 和 unit
		// prompt 是 "今天上海的气温是多少摄氏度？然后你的任务是根据今天的天气来安排鲜花的存储。"
		// genmini 会传递 location=上海 unit=今天，unit 明显传递内容有误
		parameters[paramName] = &genai.Schema{
			Type: mapGoTypeToGenaiType(paramType),
		}
		required = append(required, paramName)
	}

	gc.Functions[name] = fnValue

	functionDecl := &genai.FunctionDeclaration{
		Name:        name,
		Description: description,
		Parameters: &genai.Schema{
			Type:       genai.TypeObject,
			Properties: parameters,
			Required:   required,
		},
	}

	tool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{functionDecl},
	}
	gc.Tools = append(gc.Tools, tool)

	return nil
}

// MultiQueryWithCallbacks processes a prompt, supports function tools, and uses a callback function to handle function responses.
func (gc *GeminiClient) MultiQueryWithCallbacks(prompt string, base64Data, dataMimeType *string, temperature *float32, callback FunctionCallHandler) (string, error) {
	if strings.TrimSpace(prompt) == "" {
		return "", ErrEmptyPrompt
	}

	gc.ClearParts()
	gc.AddText(prompt)

	if base64Data != nil && dataMimeType != nil {
		data, err := base64.StdEncoding.DecodeString(*base64Data)
		if err != nil {
			return "", fmt.Errorf("failed to decode base64 data: %v", err)
		}
		gc.AddData(*dataMimeType, data)
	}

	ctx, cancel := context.WithTimeout(context.Background(), gc.Timeout)
	defer cancel()

	model := gc.Client.GenerativeModel(gc.ModelName)
	if temperature != nil {
		model.SetTemperature(*temperature)
	}
	model.Tools = gc.Tools
	session := model.StartChat()

	res, err := session.SendMessage(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to send message: %v", err)
	}

	for _, candidate := range res.Candidates {
		for _, part := range candidate.Content.Parts {
			if funcall, ok := part.(genai.FunctionCall); ok {
				fm := gc.FuncMetadataManager.Fms[funcall.Name]
				if fm == nil {
					return "", fmt.Errorf("function %s not found", funcall.Name)
				}
				responseData, err := gc.invokeFunction(funcall, fm)
				if err != nil {
					return "", fmt.Errorf("failed to handle function call: %v", err)
				}

				if callback != nil {
					responseData, err = callback(responseData)
					if err != nil {
						return "", fmt.Errorf("callback processing failed: %v", err)
					}
				}

				res, err = session.SendMessage(ctx, genai.FunctionResponse{
					Name:     funcall.Name,
					Response: responseData,
				})
				if err != nil {
					return "", fmt.Errorf("failed to send function response: %v", err)
				}

				var finalResult strings.Builder
				for _, part := range res.Candidates[0].Content.Parts {
					if textPart, ok := part.(genai.Text); ok {
						finalResult.WriteString(string(textPart))
						finalResult.WriteString("\n")
					}
				}
				return strings.TrimSpace(finalResult.String()), nil
			}
		}
	}

	result, err := gc.SubmitToClient(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to process response: %v", err)
	}

	return strings.TrimSpace(result), nil
}

// MultiQueryWithSequentialCallbacks handles multiple function calls in sequence, using callback functions to manage responses.
func (gc *GeminiClient) MultiQueryWithSequentialCallbacks(prompt string, callbacks map[string]FunctionCallHandler) (string, error) {
	if strings.TrimSpace(prompt) == "" {
		return "", ErrEmptyPrompt
	}

	gc.ClearParts()
	gc.AddText(prompt)

	ctx, cancel := context.WithTimeout(context.Background(), gc.Timeout)
	defer cancel()

	model := gc.Client.GenerativeModel(gc.ModelName)
	model.Tools = gc.Tools
	session := model.StartChat()

	res, err := session.SendMessage(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to send message: %v", err)
	}

	for _, candidate := range res.Candidates {
		for _, part := range candidate.Content.Parts {
			if funcall, ok := part.(genai.FunctionCall); ok {
				handler, exists := callbacks[funcall.Name]
				if !exists {
					return "", fmt.Errorf("no handler found for function: %s", funcall.Name)
				}

				responseData, err := handler(funcall.Args)
				if err != nil {
					return "", fmt.Errorf("handler error for function %s: %v", funcall.Name, err)
				}

				res, err = session.SendMessage(ctx, genai.FunctionResponse{
					Name:     funcall.Name,
					Response: responseData,
				})
				if err != nil {
					return "", fmt.Errorf("failed to send function response: %v", err)
				}
			}
		}
	}

	finalResult, err := gc.SubmitToClient(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to process final response: %v", err)
	}

	return strings.TrimSpace(finalResult), nil
}

// invokeFunction uses reflection to call the appropriate user-defined function based on the AI's request.
func (gc *GeminiClient) invokeFunction(fc genai.FunctionCall, fm *FuncMetadata) (map[string]any, error) {
	if fm == nil {
		return nil, fmt.Errorf("BUG: fm is nil")
	}

	// 检查 genai.FunctionCall 提供的参数是否与实际函数参数匹配
	fnType := fm.fnType
	NumParam := len(fc.Args)
	if NumParam != fnType.NumIn() {
		return nil, fmt.Errorf("function %s has %d parameters, but %d arguments were provided", fm.FnName, fnType.NumIn(), NumParam)
	}

	in := make([]reflect.Value, 0, NumParam)
	for _, v := range fc.Args {
		in = append(in, reflect.ValueOf(v))
	}

	// in := make([]reflect.Value, fnType.NumIn())
	// for i := 0; i < fnType.NumIn(); i++ {
	// 	paramName := fmt.Sprintf("param%d", i+1)
	// 	argValue, exists := args[paramName]
	// 	if !exists {
	// 		return nil, fmt.Errorf("missing argument: %s", paramName)
	// 	}
	// 	in[i] = reflect.ValueOf(argValue)
	// }

	out := fm.fnValue.Call(in)
	if len(out) < 1 {
		return nil, fmt.Errorf("function %s returned no value", fm.FnName)
	}
	if out[0].Kind() != reflect.Map {
		return nil, fmt.Errorf("function %s returned a non-map value", fm.FnName)
	}
	result := out[0].Interface().(map[string]any)
	//result := make(map[string]any)
	//for i := 0; i < len(out); i++ {
	//	// result[fmt.Sprintf("return%d", i+1)] = out[i].Interface()
	//	result[fm.RetNames[i]] = out[i].Interface()
	//}

	return result, nil
}

// ClearToolsAndFunctions clears all registered tools and functions.
func (gc *GeminiClient) ClearToolsAndFunctions() {
	gc.Functions = make(map[string]reflect.Value)
	gc.Tools = []*genai.Tool{}
}

// mapGoTypeToGenaiType maps Go types to the corresponding genai.Schema Type values.
func mapGoTypeToGenaiType(goType reflect.Type) genai.Type {
	switch goType.Kind() {
	case reflect.String:
		return genai.TypeString
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return genai.TypeInteger
	case reflect.Float32, reflect.Float64:
		return genai.TypeNumber
	case reflect.Bool:
		return genai.TypeBoolean
	default:
		return genai.TypeString
	}
}
