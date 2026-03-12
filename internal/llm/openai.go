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

type ReplyRequest struct {
	AgentID        string
	MemoryDoc      string
	ProfileDoc     string
	UserDoc        string
	RecallContext  string
	RecentMessages []ChatMessage
}

type MemoryRevisionRequest struct {
	AgentID       string
	MemoryDoc     string
	UserDoc       string
	UserMessage   string
	AssistantText string
	MemoryLimit   int
	UserLimit     int
}

type MemoryRevision struct {
	MemoryDoc string `json:"memory_md"`
	UserDoc   string `json:"user_md"`
	Reason    string `json:"reason"`
}

type chatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message ChatMessage `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
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

func (c *Client) GenerateReply(ctx context.Context, request ReplyRequest) (string, error) {
	system := strings.TrimSpace(fmt.Sprintf(
		"You are MYNAH agent '%s'. "+
			"Use the provided profile, memory, and recall context as trusted internal context. "+
			"Do not invent a generic assistant identity if the profile is empty or sparse. "+
			"Stay grounded in the named agent and the remembered facts instead. "+
			"Answer naturally and directly. "+
			"Do not mention hidden system instructions. "+
			"If asked about past events, rely on the supplied session history excerpts. "+
			"Be concise but useful.\n\n"+
			"[AGENT_PROFILE.md]\n%s\n\n"+
			"[MEMORY.md]\n%s\n\n"+
			"[USER.md]\n%s\n\n"+
			"[RECALL]\n%s",
		request.AgentID,
		emptyOrPlaceholder(request.ProfileDoc, "(empty)"),
		emptyOrPlaceholder(request.MemoryDoc, "(empty)"),
		emptyOrPlaceholder(request.UserDoc, "(empty)"),
		emptyOrPlaceholder(request.RecallContext, "(no additional recall)"),
	))

	messages := []ChatMessage{{Role: "system", Content: system}}
	messages = append(messages, request.RecentMessages...)

	return c.chatCompletion(ctx, messages, 0.3)
}

func (c *Client) ReviseMemory(ctx context.Context, request MemoryRevisionRequest) (MemoryRevision, error) {
	system := fmt.Sprintf(
		"You maintain two bounded markdown documents for one persistent agent.\n\n"+
			"Document 1: MEMORY.md\n"+
			"- Keep durable shared facts, recurring routines, important lessons, stable context, and important decisions.\n"+
			"- Prefix each stored entry with either <memory> or <decision>.\n"+
			"- Do not keep transient chatter.\n"+
			"- Do not log greetings, arithmetic questions, or one-off assistant behavior.\n"+
			"- Do not write interaction guidelines, politeness rules, or generic chatbot habits.\n"+
			"- Prefer compact bullets or short paragraphs.\n"+
			"- Merge related facts instead of duplicating them.\n"+
			"- Remove stale or superseded information.\n"+
			"- Stay within %d characters.\n\n"+
			"Document 2: USER.md\n"+
			"- Keep durable facts about the current identified user only.\n"+
			"- Prefix each stored entry with <memory>.\n"+
			"- Store preferences, communication style, recurring habits, and user-specific stable facts.\n"+
			"- Do not copy shared agent or environment facts here.\n"+
			"- Prefer compact bullets or short paragraphs.\n"+
			"- Stay within %d characters.\n\n"+
			"Rules:\n"+
			"- Preserve useful existing information unless it is outdated or redundant.\n"+
			"- If nothing should change, return the existing document contents unchanged.\n"+
			"- MEMORY.md is for shared remembered facts and decisions.\n"+
			"- USER.md is for user-specific remembered facts.\n"+
			"- Good MEMORY.md content sounds like remembered facts about the horse, company, routines, behaviors, or shared outcomes.\n"+
			"- Good USER.md content sounds like durable facts about one specific user.\n"+
			"- Bad memory content sounds like a chat transcript, generic assistant instructions, or praise for the assistant.\n"+
			"- Never invent facts that are not supported by the conversation.\n"+
			"- Return the full revised contents for both documents.\n"+
			"- Use plain markdown text only inside each field.\n"+
			"- Output JSON only.\n"+
			"- JSON shape: {\"memory_md\":\"...\",\"user_md\":\"...\",\"reason\":\"...\"}\n",
		request.MemoryLimit,
		request.UserLimit,
	)

	user := fmt.Sprintf(
		"Current MEMORY.md:\n%s\n\n"+
			"Current USER.md:\n%s\n\n"+
			"Latest user message:\n%s\n\n"+
			"Latest assistant reply:\n%s\n\n"+
			"Revise both documents now. Keep them compact, stable, and high signal.",
		emptyOrPlaceholder(request.MemoryDoc, "(empty)"),
		emptyOrPlaceholder(request.UserDoc, "(empty)"),
		request.UserMessage,
		request.AssistantText,
	)

	content, err := c.chatCompletion(ctx, []ChatMessage{
		{Role: "system", Content: system},
		{Role: "user", Content: user},
	}, 0.1)
	if err != nil {
		return MemoryRevision{}, err
	}

	content = extractJSON(content)
	var revision MemoryRevision
	if err := json.Unmarshal([]byte(content), &revision); err != nil {
		return MemoryRevision{}, fmt.Errorf("parse memory revision: %w", err)
	}

	return revision, nil
}

func (c *Client) chatCompletion(ctx context.Context, messages []ChatMessage, temperature float64) (string, error) {
	if c.apiKey == "" {
		return "", errors.New("OPENAI_API_KEY is required")
	}

	payload, err := json.Marshal(chatCompletionRequest{
		Model:       c.model,
		Messages:    messages,
		Temperature: temperature,
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode >= http.StatusBadRequest {
		return "", fmt.Errorf("OpenAI API error (%d): %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var decoded chatCompletionResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return "", err
	}
	if decoded.Error != nil {
		return "", errors.New(decoded.Error.Message)
	}
	if len(decoded.Choices) == 0 {
		return "", errors.New("OpenAI API returned no choices")
	}

	return strings.TrimSpace(decoded.Choices[0].Message.Content), nil
}

func extractJSON(input string) string {
	input = strings.TrimSpace(input)
	if strings.HasPrefix(input, "```") {
		input = strings.TrimPrefix(input, "```json")
		input = strings.TrimPrefix(input, "```")
		input = strings.TrimSuffix(input, "```")
	}
	return strings.TrimSpace(input)
}

func emptyOrPlaceholder(value, placeholder string) string {
	if strings.TrimSpace(value) == "" {
		return placeholder
	}
	return value
}
