import { useId, useRef, useState } from "react";

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function ScanSheet({
  file,
  previewUrl,
  onFile,
}: {
  file: File | null;
  previewUrl: string | null;
  onFile: (file: File) => void;
}) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function pick() {
    inputRef.current?.click();
  }

  return (
    <>
      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept="image/png,image/jpeg"
        hidden
        onChange={(event) => {
          const picked = event.target.files?.[0];
          if (picked) onFile(picked);
          event.target.value = "";
        }}
      />
      {file && previewUrl ? (
        <button type="button" className="page-thumb" onClick={pick} title="Replace the page">
          <img src={previewUrl} alt={`Scanned page: ${file.name}`} />
          <span className="page-thumb-meta">
            <span className="page-thumb-name">{file.name}</span>
            <span className="page-thumb-size">{formatSize(file.size)}</span>
          </span>
        </button>
      ) : (
        <div
          role="button"
          tabIndex={0}
          aria-label="Upload a scanned page"
          className={`dropzone${dragging ? " is-dragging" : ""}`}
          onClick={pick}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              pick();
            }
          }}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const dropped = event.dataTransfer.files?.[0];
            if (dropped && dropped.type.startsWith("image/")) onFile(dropped);
          }}
        >
          <span className="dropzone-glyph" aria-hidden="true">
            ✒
          </span>
          <p className="dropzone-title">Drop a scanned page here</p>
          <p className="dropzone-hint">or click to choose · PNG or JPEG</p>
        </div>
      )}
    </>
  );
}
