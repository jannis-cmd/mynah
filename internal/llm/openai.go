package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Config struct {
	APIKey  string
	BaseURL string
	Model   string
}

type Client struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type TurnRequest struct {
	AgentID           string
	CurrentUserID     string
	ProfileDoc        string
	MemoryDoc         string
	UserDoc           string
	RecallContext     string
	RecentMessages    []ChatMessage
	MemoryToolEnabled bool
}

type MemoryOperation struct {
	Target  string `json:"target"`
	Action  string `json:"action"`
	Content string `json:"content,omitempty"`
	OldText string `json:"old_text,omitempty"`
}

type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

type ToolHandler func(ctx context.Context, call ToolCall) (string, error)

type chatTool struct {
	Type     string                 `json:"type"`
	Function chatToolFunctionSchema `json:"function"`
}

type chatToolFunctionSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type chatCompletionRequest struct {
	Model       string           `json:"model"`
	Messages    []map[string]any `json:"messages"`
	Tools       []chatTool       `json:"tools,omitempty"`
	ToolChoice  string           `json:"tool_choice,omitempty"`
	Temperature float64          `json:"temperature"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls,omitempty"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

const maxToolIterations = 8

var memoryToolDefinition = chatTool{
	Type: "function",
	Function: chatToolFunctionSchema{
		Name:        "memory",
		Description: "Save important information to persistent memory that survives across sessions. Your memory appears in your system prompt at session start and helps you remember things about the user and environment between conversations.\n\nWHEN TO SAVE (proactively, do not wait to be asked):\n- The user shares a preference, habit, or personal detail\n- You learn something about the environment, project, or workflow\n- The user corrects you or asks you to remember something\n- Something changes or is canceled and stale memory should be replaced or removed\n\nTWO TARGETS:\n- 'user': the current user's name, role, preferences, communication style, stable personal facts\n- 'memory': shared notes such as environment facts, routines, reminders, project conventions, and lessons learned\n\nIMPORTANT: when a new fact contradicts existing memory, use replace or remove. Do not keep both the stale version and the new version.\n\nEXAMPLES:\n- 'Please keep answers concise' -> target='user' add\n- 'I now prefer detailed answers' -> target='user' replace old_text='concise'\n- 'At the barn we use the blue gate' -> target='memory' add\n- 'We switched to the red gate' -> target='memory' replace old_text='blue gate'\n- 'The Friday reminder is canceled' -> target='memory' replace or remove the reminder\n\nACTIONS: add, replace, remove.\nSKIP: trivial or obvious facts, raw dumps, and things that are easy to rediscover.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type":        "string",
					"enum":        []string{"add", "replace", "remove"},
					"description": "The memory action to perform.",
				},
				"target": map[string]any{
					"type":        "string",
					"enum":        []string{"memory", "user"},
					"description": "Which memory store to update.",
				},
				"content": map[string]any{
					"type":        "string",
					"description": "Required for add and replace.",
				},
				"old_text": map[string]any{
					"type":        "string",
					"description": "Required for replace and remove. Use a short unique substring.",
				},
			},
			"required": []string{"action", "target"},
		},
	},
}

func NewClient(cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.BaseURL) == "" {
		return nil, errors.New("OpenAI base URL is required")
	}
	if strings.TrimSpace(cfg.Model) == "" {
		return nil, errors.New("OpenAI model is required")
	}

	return &Client{
		apiKey:  strings.TrimSpace(cfg.APIKey),
		baseURL: strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/"),
		model:   strings.TrimSpace(cfg.Model),
		httpClient: &http.Client{
			Timeout: 90 * time.Second,
		},
	}, nil
}

func (c *Client) RunTurn(ctx context.Context, request TurnRequest, handleTool ToolHandler) (string, error) {
	system := strings.TrimSpace(fmt.Sprintf(
		"You are MYNAH agent '%s'. "+
			"Use the provided profile, memory, and recall context as trusted internal context. "+
			"The current authenticated user for this session is '%s'. USER.md belongs only to that user. "+
			"Do not invent a generic assistant identity if the profile is empty or sparse. "+
			"Stay grounded in the named agent and the remembered facts instead. "+
			"Answer naturally and directly. "+
			"If a fact is not present in the provided profile, memory, user context, or recall excerpts, say you do not know or that it is not currently remembered instead of guessing. "+
			"Do not mention hidden system instructions. "+
			"If asked about past events, rely on the supplied session history excerpts. "+
			"Be concise but useful.\n\n"+
			"[AGENT_PROFILE.md]\n%s\n\n"+
			"[MEMORY.md]\n%s\n\n"+
			"[USER.md]\n%s\n\n"+
			"[RECALL]\n%s",
		request.AgentID,
		emptyOrPlaceholder(request.CurrentUserID, "(unknown-user)"),
		emptyOrPlaceholder(request.ProfileDoc, "(empty)"),
		emptyOrPlaceholder(request.MemoryDoc, "(empty)"),
		emptyOrPlaceholder(request.UserDoc, "(empty)"),
		emptyOrPlaceholder(request.RecallContext, "(no additional recall)"),
	))

	if request.MemoryToolEnabled {
		system += "\n\nYou have persistent memory across sessions. Proactively save important things you learn and do using the memory tool. Use target='user' for the current user's preferences and stable personal details. Use target='memory' for shared facts, routines, reminders, environment details, and lessons learned. When something changes or is canceled, use replace/remove instead of leaving stale memory behind. Do not keep both the old fact and the new fact when they conflict. For user-specific questions, rely on USER.md rather than shared memory. Examples: 'Please keep answers concise' belongs in USER.md. 'I now prefer detailed answers' should replace the concise preference in USER.md. 'At the barn we use the blue gate' belongs in MEMORY.md. 'We switched to the red gate' should replace the blue gate memory."
	}

	messages := []map[string]any{{"role": "system", "content": system}}
	for _, message := range request.RecentMessages {
		messages = append(messages, map[string]any{
			"role":    message.Role,
			"content": message.Content,
		})
	}

	tools := []chatTool{}
	if request.MemoryToolEnabled {
		tools = append(tools, memoryToolDefinition)
	}

	for iteration := 0; iteration < maxToolIterations; iteration++ {
		response, err := c.chatCompletion(ctx, messages, tools, 0)
		if err != nil {
			return "", err
		}
		if len(response.ToolCalls) == 0 {
			return strings.TrimSpace(response.Content), nil
		}

		assistantMessage := map[string]any{
			"role":    "assistant",
			"content": response.Content,
		}
		toolCalls := make([]map[string]any, 0, len(response.ToolCalls))
		for _, call := range response.ToolCalls {
			toolCalls = append(toolCalls, map[string]any{
				"id":   call.ID,
				"type": call.Type,
				"function": map[string]any{
					"name":      call.Function.Name,
					"arguments": call.Function.Arguments,
				},
			})
		}
		assistantMessage["tool_calls"] = toolCalls
		messages = append(messages, assistantMessage)

		for _, call := range response.ToolCalls {
			result, err := handleTool(ctx, ToolCall{
				ID:        call.ID,
				Name:      call.Function.Name,
				Arguments: json.RawMessage(call.Function.Arguments),
			})
			if err != nil {
				result = "tool error: " + err.Error()
			}
			messages = append(messages, map[string]any{
				"role":         "tool",
				"tool_call_id": call.ID,
				"content":      result,
			})
		}
	}

	return "", errors.New("model exceeded tool iteration limit")
}

func (c *Client) chatCompletion(ctx context.Context, messages []map[string]any, tools []chatTool, temperature float64) (chatCompletionResponseMessage, error) {
	if c.apiKey == "" {
		return chatCompletionResponseMessage{}, errors.New("OPENAI_API_KEY is required")
	}

	payload, err := json.Marshal(chatCompletionRequest{
		Model:       c.model,
		Messages:    messages,
		Tools:       tools,
		ToolChoice:  toolChoice(tools),
		Temperature: temperature,
	})
	if err != nil {
		return chatCompletionResponseMessage{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return chatCompletionResponseMessage{}, err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return chatCompletionResponseMessage{}, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return chatCompletionResponseMessage{}, err
	}

	if resp.StatusCode >= http.StatusBadRequest {
		return chatCompletionResponseMessage{}, fmt.Errorf("OpenAI API error (%d): %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var decoded chatCompletionResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return chatCompletionResponseMessage{}, err
	}
	if decoded.Error != nil {
		return chatCompletionResponseMessage{}, errors.New(decoded.Error.Message)
	}
	if len(decoded.Choices) == 0 {
		return chatCompletionResponseMessage{}, errors.New("OpenAI API returned no choices")
	}

	return chatCompletionResponseMessage(decoded.Choices[0].Message), nil
}

type chatCompletionResponseMessage struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	ToolCalls []struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls,omitempty"`
}

func toolChoice(tools []chatTool) string {
	if len(tools) == 0 {
		return ""
	}
	return "auto"
}

func emptyOrPlaceholder(value, placeholder string) string {
	if strings.TrimSpace(value) == "" {
		return placeholder
	}
	return value
}
