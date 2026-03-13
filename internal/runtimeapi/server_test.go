package runtimeapi

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ErniConcepts/mynah/internal/app"
	"github.com/ErniConcepts/mynah/internal/storage"
)

type stubService struct {
	initCalls    []initCall
	sessionCalls []sessionCall
	chatCalls    []chatCall
	inspectCalls []inspectCall

	chatReply   string
	sessionResp app.SessionInfo
	inspectResp app.InspectResult
}

type initCall struct {
	tenantID string
	agentID  string
}

type chatCall struct {
	tenantID  string
	agentID   string
	userID    string
	sessionID string
	message   string
}

type sessionCall struct {
	tenantID string
	agentID  string
	userID   string
}

type inspectCall struct {
	tenantID string
	agentID  string
	userID   string
	limit    int
}

func (s *stubService) InitAgent(tenantID, agentID string) error {
	s.initCalls = append(s.initCalls, initCall{tenantID: tenantID, agentID: agentID})
	return nil
}

func (s *stubService) StartSession(tenantID, agentID, userID string) (app.SessionInfo, error) {
	s.sessionCalls = append(s.sessionCalls, sessionCall{tenantID: tenantID, agentID: agentID, userID: userID})
	return s.sessionResp, nil
}

func (s *stubService) ChatOnce(_ context.Context, tenantID, agentID, userID, sessionID, userInput string) (string, error) {
	s.chatCalls = append(s.chatCalls, chatCall{
		tenantID:  tenantID,
		agentID:   agentID,
		userID:    userID,
		sessionID: sessionID,
		message:   userInput,
	})
	return s.chatReply, nil
}

func (s *stubService) InspectAgent(tenantID, agentID, userID string, messageLimit int) (app.InspectResult, error) {
	s.inspectCalls = append(s.inspectCalls, inspectCall{
		tenantID: tenantID,
		agentID:  agentID,
		userID:   userID,
		limit:    messageLimit,
	})
	return s.inspectResp, nil
}

func TestHealthz(t *testing.T) {
	handler := NewHandler(&stubService{}, 5*time.Second)
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", recorder.Code)
	}
	if !strings.Contains(recorder.Body.String(), `"status":"ok"`) {
		t.Fatalf("unexpected body: %s", recorder.Body.String())
	}
}

func TestInitAgentEndpoint(t *testing.T) {
	service := &stubService{}
	handler := NewHandler(service, 5*time.Second)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/agents/init", bytes.NewBufferString(`{"tenant_id":"demo","agent_id":"bella"}`))
	request.Header.Set("Content-Type", "application/json")

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", recorder.Code)
	}
	if len(service.initCalls) != 1 || service.initCalls[0].tenantID != "demo" || service.initCalls[0].agentID != "bella" {
		t.Fatalf("unexpected init calls: %+v", service.initCalls)
	}
}

func TestChatEndpointSuccess(t *testing.T) {
	service := &stubService{chatReply: "Stored."}
	handler := NewHandler(service, 5*time.Second)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat", bytes.NewBufferString(`{"tenant_id":"demo","agent_id":"bella","user_id":"anna","session_id":"sess_1","message":"hello"}`))
	request.Header.Set("Content-Type", "application/json")

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", recorder.Code, recorder.Body.String())
	}
	var response chatResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.SessionID != "sess_1" || response.Reply != "Stored." {
		t.Fatalf("unexpected response: %+v", response)
	}
	if len(service.chatCalls) != 1 || service.chatCalls[0].userID != "anna" {
		t.Fatalf("unexpected chat calls: %+v", service.chatCalls)
	}
}

func TestStartSessionEndpoint(t *testing.T) {
	service := &stubService{
		sessionResp: app.SessionInfo{
			TenantID:  "demo",
			AgentID:   "bella",
			UserID:    "anna",
			SessionID: "sess_123",
		},
	}
	handler := NewHandler(service, 5*time.Second)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/sessions", bytes.NewBufferString(`{"tenant_id":"demo","agent_id":"bella","user_id":"anna"}`))
	request.Header.Set("Content-Type", "application/json")

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", recorder.Code, recorder.Body.String())
	}
	if len(service.sessionCalls) != 1 || service.sessionCalls[0].userID != "anna" {
		t.Fatalf("unexpected session calls: %+v", service.sessionCalls)
	}
	if !strings.Contains(recorder.Body.String(), `"session_id":"sess_123"`) {
		t.Fatalf("unexpected body: %s", recorder.Body.String())
	}
}

func TestChatEndpointValidationFailure(t *testing.T) {
	handler := NewHandler(&stubService{}, 5*time.Second)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat", bytes.NewBufferString(`{"tenant_id":"demo","agent_id":"bella","message":"hello"}`))
	request.Header.Set("Content-Type", "application/json")

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", recorder.Code)
	}
	if !strings.Contains(recorder.Body.String(), "user_id") {
		t.Fatalf("unexpected body: %s", recorder.Body.String())
	}
}

func TestChatEndpointRequiresSessionID(t *testing.T) {
	handler := NewHandler(&stubService{}, 5*time.Second)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat", bytes.NewBufferString(`{"tenant_id":"demo","agent_id":"bella","user_id":"anna","message":"hello"}`))
	request.Header.Set("Content-Type", "application/json")

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", recorder.Code)
	}
	if !strings.Contains(recorder.Body.String(), "/v1/sessions") {
		t.Fatalf("unexpected body: %s", recorder.Body.String())
	}
}

func TestInspectEndpoint(t *testing.T) {
	service := &stubService{
		inspectResp: app.InspectResult{
			TenantID:   "demo",
			AgentID:    "bella",
			UserID:     "anna",
			AgentRoot:  "C:\\demo",
			MemoryDoc:  "The barn uses the blue gate.",
			ProfileDoc: "Bella profile",
			UserDoc:    "Prefers concise answers.",
			RecentMessages: []storage.Message{
				{Role: "user", Content: "hello"},
			},
		},
	}
	handler := NewHandler(service, 5*time.Second)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/inspect?tenant_id=demo&agent_id=bella&user_id=anna&limit=5", nil)

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", recorder.Code, recorder.Body.String())
	}
	if len(service.inspectCalls) != 1 || service.inspectCalls[0].limit != 5 {
		t.Fatalf("unexpected inspect calls: %+v", service.inspectCalls)
	}
	if !strings.Contains(recorder.Body.String(), "blue gate") {
		t.Fatalf("unexpected body: %s", recorder.Body.String())
	}
}

func TestUnknownRoute(t *testing.T) {
	handler := NewHandler(&stubService{}, 5*time.Second)
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/unknown", nil)

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", recorder.Code)
	}
}
