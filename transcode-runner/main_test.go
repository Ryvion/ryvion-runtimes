package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestRunWritesSuccessContractAndRootArtifact(t *testing.T) {
	workRoot := t.TempDir()
	writeJob(t, workRoot, JobSpec{
		InputFile:  "input.mov",
		OutputFile: "nested/final.mp4",
	})
	if err := os.WriteFile(filepath.Join(workRoot, "input.mov"), []byte("input"), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}
	withFakeCommand(t, "ffmpeg", `#!/bin/sh
out=""
for arg in "$@"; do
  out="$arg"
done
printf 'video-output' > "$out"
`)

	if err := run(workRoot); err != nil {
		t.Fatalf("run() error = %v", err)
	}

	receipt := readJSONFile[receipt](t, filepath.Join(workRoot, "receipt.json"))
	if !receipt.OK || receipt.OutputHash == "" {
		t.Fatalf("unexpected receipt: %+v", receipt)
	}
	metrics := readJSONFile[map[string]any](t, filepath.Join(workRoot, "metrics.json"))
	if metrics["output_name"] != "final.mp4" {
		t.Fatalf("output_name = %v, want final.mp4", metrics["output_name"])
	}
	if _, err := os.Stat(filepath.Join(workRoot, "final.mp4")); err != nil {
		t.Fatalf("root artifact missing: %v", err)
	}
}

func TestRunMissingInputWritesFailureContract(t *testing.T) {
	workRoot := t.TempDir()
	writeJob(t, workRoot, JobSpec{InputFile: "missing.mov"})

	if err := run(workRoot); err == nil {
		t.Fatal("run() error = nil, want missing input error")
	}

	receipt := readJSONFile[receipt](t, filepath.Join(workRoot, "receipt.json"))
	if receipt.OK || receipt.OutputHash == "" || !strings.Contains(receipt.Error, "input file missing") {
		t.Fatalf("unexpected failure receipt: %+v", receipt)
	}
	metrics := readJSONFile[map[string]any](t, filepath.Join(workRoot, "metrics.json"))
	if metrics["output_name"] != "output.mp4" {
		t.Fatalf("output_name = %v, want output.mp4", metrics["output_name"])
	}
	if metrics["error_stage"] != "input_missing" {
		t.Fatalf("error_stage = %v, want input_missing", metrics["error_stage"])
	}
}

func TestRunEscapedPathWritesFailureContract(t *testing.T) {
	workRoot := t.TempDir()
	writeJob(t, workRoot, JobSpec{InputFile: "input.mov", OutputFile: "../escape.mp4"})
	if err := os.WriteFile(filepath.Join(workRoot, "input.mov"), []byte("input"), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	if err := run(workRoot); err == nil {
		t.Fatal("run() error = nil, want path escape error")
	}

	receipt := readJSONFile[receipt](t, filepath.Join(workRoot, "receipt.json"))
	if receipt.OK || receipt.OutputHash == "" {
		t.Fatalf("unexpected failure receipt: %+v", receipt)
	}
	metrics := readJSONFile[map[string]any](t, filepath.Join(workRoot, "metrics.json"))
	if metrics["output_name"] != "escape.mp4" {
		t.Fatalf("output_name = %v, want escape.mp4", metrics["output_name"])
	}
	if metrics["error_stage"] != "resolve_paths" {
		t.Fatalf("error_stage = %v, want resolve_paths", metrics["error_stage"])
	}
}

func TestRunFFmpegFailureWritesFailureContract(t *testing.T) {
	workRoot := t.TempDir()
	writeJob(t, workRoot, JobSpec{InputFile: "input.mov", OutputFile: "failed.mp4"})
	if err := os.WriteFile(filepath.Join(workRoot, "input.mov"), []byte("input"), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}
	withFakeCommand(t, "ffmpeg", `#!/bin/sh
echo 'codec failed' >&2
exit 7
`)

	if err := run(workRoot); err == nil {
		t.Fatal("run() error = nil, want ffmpeg error")
	}

	receipt := readJSONFile[receipt](t, filepath.Join(workRoot, "receipt.json"))
	if receipt.OK || receipt.OutputHash == "" || !strings.Contains(receipt.Error, "codec failed") {
		t.Fatalf("unexpected failure receipt: %+v", receipt)
	}
	metrics := readJSONFile[map[string]any](t, filepath.Join(workRoot, "metrics.json"))
	if metrics["output_name"] != "failed.mp4" {
		t.Fatalf("output_name = %v, want failed.mp4", metrics["output_name"])
	}
	if metrics["error_stage"] != "ffmpeg" {
		t.Fatalf("error_stage = %v, want ffmpeg", metrics["error_stage"])
	}
}

func writeJob(t *testing.T, workRoot string, spec JobSpec) {
	t.Helper()
	data, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("marshal job: %v", err)
	}
	if err := os.WriteFile(filepath.Join(workRoot, "job.json"), data, 0o644); err != nil {
		t.Fatalf("write job: %v", err)
	}
}

func withFakeCommand(t *testing.T, name, script string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fake command is POSIX-only")
	}
	binDir := t.TempDir()
	path := filepath.Join(binDir, name)
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("write fake %s: %v", name, err)
	}
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
}

func readJSONFile[T any](t *testing.T, path string) T {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var out T
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
	return out
}
