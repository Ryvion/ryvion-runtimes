package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type JobSpec struct {
	InputFile  string   `json:"input_file"`
	OutputFile string   `json:"output_file"`
	Args       []string `json:"args"`
}

type receipt struct {
	OutputHash string `json:"output_hash"`
	OK         bool   `json:"ok"`
	Error      string `json:"error,omitempty"`
}

type ffprobeOutput struct {
	Streams []ffprobeStream `json:"streams"`
	Format  ffprobeFormat   `json:"format"`
}

type ffprobeStream struct {
	CodecType    string `json:"codec_type"`
	CodecName    string `json:"codec_name"`
	Width        int    `json:"width"`
	Height       int    `json:"height"`
	AvgFrameRate string `json:"avg_frame_rate"`
	Channels     int    `json:"channels"`
	SampleRate   string `json:"sample_rate"`
}

type ffprobeFormat struct {
	FormatName string `json:"format_name"`
	Duration   string `json:"duration"`
	Size       string `json:"size"`
	BitRate    string `json:"bit_rate"`
}

func main() {
	if err := run("/work"); err != nil {
		log.Print(err)
		os.Exit(1)
	}
}

func run(workRoot string) error {
	start := time.Now()
	js, err := loadJob(filepath.Join(workRoot, "job.json"))
	if err != nil {
		return failRun(workRoot, JobSpec{}, "load_job", err, time.Since(start))
	}

	inputPath, outputPath, err := resolvePaths(workRoot, js)
	if err != nil {
		return failRun(workRoot, js, "resolve_paths", err, time.Since(start))
	}
	if err := requireInput(inputPath); err != nil {
		return failRun(workRoot, js, "input_missing", err, time.Since(start))
	}

	args := buildArgs(js, inputPath, outputPath)
	cmd := exec.Command("ffmpeg", args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		if len(out) > 0 {
			err = fmt.Errorf("ffmpeg: %w: %s", err, strings.TrimSpace(string(out)))
		} else {
			err = fmt.Errorf("ffmpeg: %w", err)
		}
		return failRun(workRoot, js, "ffmpeg", err, time.Since(start))
	}
	durationMs := time.Since(start).Milliseconds()

	artifactPath, err := ensureRootArtifact(workRoot, outputPath)
	if err != nil {
		return failRun(workRoot, js, "artifact", err, time.Since(start))
	}
	sum, size, err := hashFile(artifactPath)
	if err != nil {
		return failRun(workRoot, js, "hash_output", err, time.Since(start))
	}
	if err := writeJSON(filepath.Join(workRoot, "receipt.json"), receipt{OutputHash: sum, OK: true}); err != nil {
		return err
	}
	metrics := map[string]any{
		"engine":       "ffmpeg",
		"duration_ms":  durationMs,
		"output_bytes": size,
		"output_name":  filepath.Base(artifactPath),
	}
	if ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(artifactPath)), "."); ext != "" {
		metrics["output_format"] = ext
	}
	for key, value := range probeOutput(artifactPath) {
		metrics[key] = value
	}
	if err := writeJSON(filepath.Join(workRoot, "metrics.json"), metrics); err != nil {
		return err
	}
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"output_hash": sum,
		"ok":          true,
	})
	return nil
}

func loadJob(path string) (JobSpec, error) {
	f, err := os.Open(path)
	if err != nil {
		return JobSpec{}, fmt.Errorf("open job spec: %w", err)
	}
	defer f.Close()

	var js JobSpec
	if err := json.NewDecoder(f).Decode(&js); err != nil {
		return JobSpec{}, fmt.Errorf("decode job spec: %w", err)
	}
	return js, nil
}

func resolvePaths(workRoot string, js JobSpec) (string, string, error) {
	inputPath, err := safeWorkPath(workRoot, js.InputFile, "input")
	if err != nil {
		return "", "", err
	}
	outputPath, err := safeWorkPath(workRoot, js.OutputFile, "output.mp4")
	if err != nil {
		return "", "", err
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return "", "", fmt.Errorf("create output directory: %w", err)
	}
	return inputPath, outputPath, nil
}

func safeWorkPath(workRoot, value, fallback string) (string, error) {
	root := filepath.Clean(workRoot)
	raw := strings.TrimSpace(value)
	if raw == "" {
		raw = fallback
	}
	if !filepath.IsAbs(raw) {
		raw = filepath.Join(root, raw)
	}
	clean := filepath.Clean(raw)
	rel, err := filepath.Rel(root, clean)
	if err != nil || rel == ".." || strings.HasPrefix(rel, "../") || filepath.IsAbs(rel) {
		return "", fmt.Errorf("path escapes %s: %s", root, value)
	}
	return clean, nil
}

func requireInput(inputPath string) error {
	if _, err := os.Stat(inputPath); err != nil {
		return fmt.Errorf("input file missing: %w", err)
	}
	return nil
}

func buildArgs(js JobSpec, inputPath, outputPath string) []string {
	if len(js.Args) == 0 {
		return []string{
			"-y",
			"-i", inputPath,
			"-c:v", "libx264",
			"-preset", "veryfast",
			"-movflags", "+faststart",
			"-pix_fmt", "yuv420p",
			"-c:a", "aac",
			"-b:a", "128k",
			"-f", "mp4",
			outputPath,
		}
	}
	args := make([]string, 0, len(js.Args))
	for _, a := range js.Args {
		a = strings.TrimSpace(a)
		switch a {
		case "{input}":
			a = inputPath
		case "{output}":
			a = outputPath
		}
		args = append(args, a)
	}
	return args
}

func hashFile(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, fmt.Errorf("open output: %w", err)
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, fmt.Errorf("hash output: %w", err)
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func ensureRootArtifact(workRoot, outputPath string) (string, error) {
	rootArtifact := filepath.Join(filepath.Clean(workRoot), filepath.Base(outputPath))
	if filepath.Clean(outputPath) == rootArtifact {
		return outputPath, nil
	}
	in, err := os.Open(outputPath)
	if err != nil {
		return "", fmt.Errorf("open output artifact: %w", err)
	}
	defer in.Close()
	out, err := os.Create(rootArtifact)
	if err != nil {
		return "", fmt.Errorf("create root artifact: %w", err)
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		return "", fmt.Errorf("copy root artifact: %w", err)
	}
	if err := out.Close(); err != nil {
		return "", fmt.Errorf("close root artifact: %w", err)
	}
	return rootArtifact, nil
}

func failRun(workRoot string, js JobSpec, stage string, runErr error, duration time.Duration) error {
	outputName := safeOutputName(js.OutputFile)
	errorText := strings.TrimSpace(runErr.Error())
	failureHash := failureOutputHash(stage, errorText)
	writeErr := writeFailureContract(workRoot, outputName, stage, errorText, failureHash, duration)
	if writeErr != nil {
		return errors.Join(runErr, writeErr)
	}
	return runErr
}

func writeFailureContract(workRoot, outputName, stage, errorText, outputHash string, duration time.Duration) error {
	if err := writeJSON(filepath.Join(workRoot, "receipt.json"), receipt{
		OutputHash: outputHash,
		OK:         false,
		Error:      errorText,
	}); err != nil {
		return err
	}
	return writeJSON(filepath.Join(workRoot, "metrics.json"), map[string]any{
		"engine":      "ffmpeg",
		"duration_ms": duration.Milliseconds(),
		"output_name": outputName,
		"status":      "failed",
		"error_stage": stage,
		"error":       errorText,
	})
}

func failureOutputHash(stage, errorText string) string {
	sum := sha256.Sum256([]byte("transcode-runner failure\n" + stage + "\n" + errorText))
	return hex.EncodeToString(sum[:])
}

func safeOutputName(outputFile string) string {
	name := filepath.Base(filepath.Clean(strings.TrimSpace(outputFile)))
	if name == "." || name == string(filepath.Separator) || name == "" {
		return "output.mp4"
	}
	return name
}

func probeOutput(path string) map[string]any {
	cmd := exec.Command(
		"ffprobe",
		"-v", "error",
		"-print_format", "json",
		"-show_format",
		"-show_streams",
		path,
	)
	out, err := cmd.Output()
	if err != nil {
		return map[string]any{
			"probe_error": err.Error(),
		}
	}

	var probe ffprobeOutput
	if err := json.Unmarshal(out, &probe); err != nil {
		return map[string]any{
			"probe_error": fmt.Sprintf("decode ffprobe: %v", err),
		}
	}

	metrics := map[string]any{}
	if name := strings.TrimSpace(probe.Format.FormatName); name != "" {
		metrics["container_family"] = firstCSVToken(name)
	}
	if seconds, ok := parseFloat(probe.Format.Duration); ok && seconds > 0 {
		metrics["output_duration_seconds"] = seconds
	}
	if bitrate, ok := parseInt(probe.Format.BitRate); ok && bitrate > 0 {
		metrics["output_bitrate_bps"] = bitrate
	}
	for _, stream := range probe.Streams {
		switch strings.TrimSpace(stream.CodecType) {
		case "video":
			if codec := strings.TrimSpace(stream.CodecName); codec != "" {
				metrics["video_codec"] = codec
			}
			if stream.Width > 0 {
				metrics["width"] = stream.Width
			}
			if stream.Height > 0 {
				metrics["height"] = stream.Height
			}
			if fps, ok := parseFrameRate(stream.AvgFrameRate); ok && fps > 0 {
				metrics["frame_rate"] = fps
			}
		case "audio":
			if codec := strings.TrimSpace(stream.CodecName); codec != "" {
				metrics["audio_codec"] = codec
			}
			if stream.Channels > 0 {
				metrics["audio_channels"] = stream.Channels
			}
			if sampleRate, ok := parseInt(stream.SampleRate); ok && sampleRate > 0 {
				metrics["audio_sample_rate_hz"] = sampleRate
			}
		}
	}
	return metrics
}

func firstCSVToken(value string) string {
	if idx := strings.Index(value, ","); idx >= 0 {
		return strings.TrimSpace(value[:idx])
	}
	return strings.TrimSpace(value)
}

func parseFloat(value string) (float64, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, false
	}
	return parsed, true
}

func parseInt(value string) (int64, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return 0, false
	}
	return parsed, true
}

func parseFrameRate(value string) (float64, bool) {
	value = strings.TrimSpace(value)
	if value == "" || value == "0/0" {
		return 0, false
	}
	if strings.Contains(value, "/") {
		parts := strings.SplitN(value, "/", 2)
		num, okNum := parseFloat(parts[0])
		den, okDen := parseFloat(parts[1])
		if !okNum || !okDen || den == 0 {
			return 0, false
		}
		return num / den, true
	}
	return parseFloat(value)
}

func writeJSON(path string, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal %s: %w", path, err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

func init() {
	log.SetFlags(0)
	log.SetPrefix("transcode-runner: ")
}
