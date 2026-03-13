package runtimeapi

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/ErniConcepts/mynah/internal/app"
)

type AppService interface {
	InitAgent(tenantID, agentID string) error
	StartSession(tenantID, agentID, userID string, channel app.ChannelInfo) (app.SessionInfo, error)
	ChatOnce(ctx context.Context, tenantID, agentID, userID, sessionID, userInput string) (string, error)
	InspectAgent(tenantID, agentID, userID string, messageLimit int) (app.InspectResult, error)
}

type Server struct {
	service        AppService
	defaultTimeout time.Duration
}

type initAgentRequest struct {
	TenantID string `json:"tenant_id"`
	AgentID  string `json:"agent_id"`
}

type chatRequest struct {
	TenantID  string `json:"tenant_id"`
	AgentID   string `json:"agent_id"`
	UserID    string `json:"user_id"`
	SessionID string `json:"session_id"`
	Message   string `json:"message"`
	TimeoutMS int    `json:"timeout_ms,omitempty"`
}

type startSessionRequest struct {
	TenantID string `json:"tenant_id"`
	AgentID  string `json:"agent_id"`
	UserID   string `json:"user_id"`
	Channel  struct {
		Type    string `json:"type,omitempty"`
		Subject string `json:"subject,omitempty"`
	} `json:"channel,omitempty"`
}

type chatResponse struct {
	SessionID string `json:"session_id"`
	Reply     string `json:"reply"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func NewHandler(service AppService, defaultTimeout time.Duration) http.Handler {
	server := &Server{
		service:        service,
		defaultTimeout: defaultTimeout,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", server.handleHealthz)
	mux.HandleFunc("POST /v1/agents/init", server.handleInitAgent)
	mux.HandleFunc("POST /v1/sessions", server.handleStartSession)
	mux.HandleFunc("POST /v1/chat", server.handleChat)
	mux.HandleFunc("GET /v1/inspect", server.handleInspect)
	return mux
}

func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleInitAgent(w http.ResponseWriter, r *http.Request) {
	var request initAgentRequest
	if err := decodeJSON(r, &request); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if strings.TrimSpace(request.TenantID) == "" || strings.TrimSpace(request.AgentID) == "" {
		writeError(w, http.StatusBadRequest, errors.New("tenant_id and agent_id are required"))
		return
	}
	if err := s.service.InitAgent(request.TenantID, request.AgentID); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{
		"tenant_id": request.TenantID,
		"agent_id":  request.AgentID,
		"status":    "initialized",
	})
}

func (s *Server) handleStartSession(w http.ResponseWriter, r *http.Request) {
	var request startSessionRequest
	if err := decodeJSON(r, &request); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if strings.TrimSpace(request.TenantID) == "" || strings.TrimSpace(request.AgentID) == "" || strings.TrimSpace(request.UserID) == "" {
		writeError(w, http.StatusBadRequest, errors.New("tenant_id, agent_id, and user_id are required"))
		return
	}

	session, err := s.service.StartSession(request.TenantID, request.AgentID, request.UserID, app.ChannelInfo{
		Type:    strings.TrimSpace(request.Channel.Type),
		Subject: strings.TrimSpace(request.Channel.Subject),
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, session)
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var request chatRequest
	if err := decodeJSON(r, &request); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if strings.TrimSpace(request.TenantID) == "" || strings.TrimSpace(request.AgentID) == "" || strings.TrimSpace(request.UserID) == "" || strings.TrimSpace(request.Message) == "" {
		writeError(w, http.StatusBadRequest, errors.New("tenant_id, agent_id, user_id, and message are required"))
		return
	}

	sessionID := strings.TrimSpace(request.SessionID)
	if sessionID == "" {
		writeError(w, http.StatusBadRequest, errors.New("session_id is required; create one with POST /v1/sessions"))
		return
	}

	ctx := r.Context()
	timeout := s.defaultTimeout
	if request.TimeoutMS > 0 {
		timeout = time.Duration(request.TimeoutMS) * time.Millisecond
	}
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	reply, err := s.service.ChatOnce(ctx, request.TenantID, request.AgentID, request.UserID, sessionID, request.Message)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	writeJSON(w, http.StatusOK, chatResponse{
		SessionID: sessionID,
		Reply:     reply,
	})
}

func (s *Server) handleInspect(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	tenantID := strings.TrimSpace(query.Get("tenant_id"))
	agentID := strings.TrimSpace(query.Get("agent_id"))
	userID := strings.TrimSpace(query.Get("user_id"))
	if tenantID == "" || agentID == "" {
		writeError(w, http.StatusBadRequest, errors.New("tenant_id and agent_id are required"))
		return
	}

	limit := 12
	if rawLimit := strings.TrimSpace(query.Get("limit")); rawLimit != "" {
		parsed, err := strconv.Atoi(rawLimit)
		if err != nil || parsed < 0 {
			writeError(w, http.StatusBadRequest, errors.New("limit must be a non-negative integer"))
			return
		}
		limit = parsed
	}

	result, err := s.service.InspectAgent(tenantID, agentID, userID, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func decodeJSON(r *http.Request, target any) error {
	defer r.Body.Close()
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	return nil
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, errorResponse{Error: err.Error()})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	payload, err := json.Marshal(value)
	if err != nil {
		http.Error(w, `{"error":"internal json error"}`, http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(payload)
}
