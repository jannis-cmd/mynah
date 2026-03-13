package storage

import "testing"

func TestBuildFTSQueryStripsPunctuationIntoSafeTokens(t *testing.T) {
	query := buildFTSQuery("Do we remember carrots, apples, and hay?")
	if query != `"carrots" "apples" "hay"` {
		t.Fatalf("unexpected query: %q", query)
	}
}

func TestBuildFTSQueryNeutralizesFTSOperators(t *testing.T) {
	query := buildFTSQuery(`warmup OR gallop NOT jump*`)
	if query != `"warmup" "OR" "gallop" "NOT" "jump"` {
		t.Fatalf("unexpected query: %q", query)
	}
}

func TestBuildFTSQueryDropsRecallBoilerplateWords(t *testing.T) {
	query := buildFTSQuery("Remember blue gate before?")
	if query != `"blue" "gate"` {
		t.Fatalf("unexpected query: %q", query)
	}
}

func TestBuildFTSQueryEmptyWhenNoTokens(t *testing.T) {
	query := buildFTSQuery(" , + {} () ^ ")
	if query != "" {
		t.Fatalf("expected empty query, got %q", query)
	}
}

func TestNewSessionStoreSetsBusyTimeout(t *testing.T) {
	store, err := NewSessionStore("file::memory:?cache=shared")
	if err != nil {
		t.Fatalf("new session store: %v", err)
	}
	defer store.Close()

	var timeoutMS int
	if err := store.db.QueryRow(`PRAGMA busy_timeout;`).Scan(&timeoutMS); err != nil {
		t.Fatalf("read busy_timeout pragma: %v", err)
	}
	if timeoutMS != 3000 {
		t.Fatalf("expected busy_timeout=3000, got %d", timeoutMS)
	}
}

func TestEnsureSessionForUserWithMetadataPersistsSourceFields(t *testing.T) {
	store, err := NewSessionStore("file::memory:?cache=shared")
	if err != nil {
		t.Fatalf("new session store: %v", err)
	}
	defer store.Close()

	if err := store.EnsureSchema(); err != nil {
		t.Fatalf("ensure schema: %v", err)
	}
	if err := store.EnsureSessionForUserWithMetadata("sess_1", "anna", SessionMetadata{
		SourceType:       "chat_platform",
		SourceSubject:    "telegram:user:12345",
		SourceSessionRef: "telegram:chat:777",
	}); err != nil {
		t.Fatalf("ensure session with metadata: %v", err)
	}

	metadata, err := store.SessionMetadata("sess_1")
	if err != nil {
		t.Fatalf("read session metadata: %v", err)
	}
	if metadata.SourceType != "chat_platform" || metadata.SourceSubject != "telegram:user:12345" || metadata.SourceSessionRef != "telegram:chat:777" {
		t.Fatalf("unexpected metadata: %+v", metadata)
	}
}

func TestEnsureSchemaMigratesLegacyChannelMetadataIntoSourceFields(t *testing.T) {
	store, err := NewSessionStore("file::memory:?cache=shared")
	if err != nil {
		t.Fatalf("new session store: %v", err)
	}
	defer store.Close()

	if _, err := store.db.Exec(`
CREATE TABLE sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  channel_type TEXT NOT NULL DEFAULT '',
  channel_subject TEXT NOT NULL DEFAULT '',
  started_at TEXT NOT NULL
);`); err != nil {
		t.Fatalf("create legacy sessions table: %v", err)
	}
	if _, err := store.db.Exec(`INSERT INTO sessions(id, user_id, channel_type, channel_subject, started_at) VALUES('sess_legacy', 'anna', 'web', 'employee_42', '2026-03-13T12:00:00Z')`); err != nil {
		t.Fatalf("insert legacy session: %v", err)
	}

	if err := store.ensureSessionColumns(); err != nil {
		t.Fatalf("ensure session columns: %v", err)
	}

	metadata, err := store.SessionMetadata("sess_legacy")
	if err != nil {
		t.Fatalf("read migrated session metadata: %v", err)
	}
	if metadata.SourceType != "web" || metadata.SourceSubject != "employee_42" {
		t.Fatalf("unexpected migrated metadata: %+v", metadata)
	}
}
