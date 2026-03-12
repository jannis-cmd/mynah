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
