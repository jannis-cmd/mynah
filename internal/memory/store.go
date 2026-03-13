package memory

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ErniConcepts/mynah/internal/llm"
	"github.com/ErniConcepts/mynah/internal/storage"
)

type Store struct {
	fileStore      *storage.FileStore
	userID         string
	memoryLimit    int
	userLimit      int
	memoryEntries  []string
	userEntries    []string
	memorySnapshot string
	userSnapshot   string
}

func NewStore(fileStore *storage.FileStore, userID string, memoryLimit, userLimit int) (*Store, error) {
	memoryDoc, err := fileStore.ReadMemory()
	if err != nil {
		return nil, err
	}
	userDoc, err := fileStore.ReadUserProfile(userID)
	if err != nil {
		return nil, err
	}

	return &Store{
		fileStore:      fileStore,
		userID:         strings.TrimSpace(userID),
		memoryLimit:    memoryLimit,
		userLimit:      userLimit,
		memoryEntries:  dedupeEntries(documentLines(memoryDoc)),
		userEntries:    dedupeEntries(documentLines(userDoc)),
		memorySnapshot: strings.TrimSpace(memoryDoc),
		userSnapshot:   strings.TrimSpace(userDoc),
	}, nil
}

func (s *Store) MemorySnapshot() string {
	return s.memorySnapshot
}

func (s *Store) UserSnapshot() string {
	return s.userSnapshot
}

func (s *Store) Execute(operation llm.MemoryOperation) (string, error) {
	target := strings.ToLower(strings.TrimSpace(operation.Target))
	action := strings.ToLower(strings.TrimSpace(operation.Action))
	content := strings.TrimSpace(operation.Content)
	oldText := strings.TrimSpace(operation.OldText)

	if target != "memory" && target != "user" {
		return s.errorResponse(fmt.Sprintf("Invalid target %q. Use 'memory' or 'user'.", operation.Target))
	}
	if action != "add" && action != "replace" && action != "remove" {
		return s.errorResponse(fmt.Sprintf("Unknown action %q. Use add, replace, or remove.", operation.Action))
	}
	if action == "add" && content == "" {
		return s.errorResponse("content is required for add")
	}
	if action == "replace" && (content == "" || oldText == "") {
		return s.errorResponse("content and old_text are required for replace")
	}
	if action == "remove" && oldText == "" {
		return s.errorResponse("old_text is required for remove")
	}

	if content != "" {
		if err := validateBase(content, 100000); err != nil {
			return s.errorResponse(err.Error())
		}
	}

	entries := &s.memoryEntries
	limit := s.memoryLimit
	if target == "user" {
		entries = &s.userEntries
		limit = s.userLimit
	}

	switch action {
	case "add":
		for _, atomicEntry := range splitAtomicEntries(content) {
			*entries = appendUniqueEntry(*entries, atomicEntry)
		}
	case "replace":
		nextEntries, err := replaceEntry(*entries, oldText, splitAtomicEntries(content))
		if err != nil {
			return s.errorResponse(err.Error())
		}
		*entries = nextEntries
	case "remove":
		nextEntries, err := removeEntry(*entries, oldText)
		if err != nil {
			return s.errorResponse(err.Error())
		}
		*entries = nextEntries
	}

	rendered := renderDocument(*entries)
	if len(rendered) > limit {
		return s.limitErrorResponse(target, len(rendered), limit, *entries)
	}

	if target == "memory" {
		if err := s.fileStore.WriteMemory(rendered); err != nil {
			return "", err
		}
	} else {
		if err := s.fileStore.WriteUserProfile(s.userID, rendered); err != nil {
			return "", err
		}
	}

	return s.successResponse(target, action, *entries)
}

func (s *Store) successResponse(target, action string, entries []string) (string, error) {
	limit := s.memoryLimit
	if target == "user" {
		limit = s.userLimit
	}
	current := len(renderDocument(entries))
	response := map[string]any{
		"success":     true,
		"target":      target,
		"action":      action,
		"entries":     entries,
		"usage":       fmt.Sprintf("%d/%d chars", current, limit),
		"entry_count": len(entries),
	}
	return marshalResponse(response)
}

func (s *Store) errorResponse(message string) (string, error) {
	return marshalResponse(map[string]any{
		"success": false,
		"error":   message,
	})
}

func (s *Store) limitErrorResponse(target string, current, limit int, entries []string) (string, error) {
	return marshalResponse(map[string]any{
		"success":         false,
		"target":          target,
		"error":           fmt.Sprintf("Memory at %d/%d chars would exceed the limit. Replace or remove existing entries first.", current, limit),
		"current_entries": entries,
		"usage":           fmt.Sprintf("%d/%d chars", current, limit),
	})
}

func marshalResponse(value any) (string, error) {
	payload, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	return string(payload), nil
}
