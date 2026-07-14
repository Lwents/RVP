import { Check } from "lucide-react";
import type { ProcessingMode } from "../types/api";

const processingModes: Array<{
  label: string;
  value: ProcessingMode;
  description: string;
}> = [
  { label: "Siêu nhanh", value: "fast", description: "Ưu tiên tốc độ, giảm các bước phân tích và xử lý nặng." },
  { label: "Cân bằng", value: "balanced", description: "Cân đối thời gian với chất lượng, phù hợp cấu hình máy hiện tại." },
  { label: "Đẹp nhất", value: "quality", description: "Ưu tiên chất lượng hình và phân tích kỹ hơn, thời gian chạy lâu hơn." },
];

interface ProcessingModeSelectorProps {
  value: ProcessingMode;
  onChange: (value: ProcessingMode) => void;
}

export function ProcessingModeSelector({ value, onChange }: ProcessingModeSelectorProps) {
  const selected = processingModes.find((item) => item.value === value) ?? processingModes[1];

  return (
    <div className="segment-wrap processing-mode-picker">
      <span className="label">Chế độ xử lý</span>
      <div className="segments processing-mode-segments" role="radiogroup" aria-label="Chế độ xử lý video">
        {processingModes.map((item) => {
          const isSelected = value === item.value;
          return (
            <button
              key={item.value}
              type="button"
              className={isSelected ? "active" : ""}
              role="radio"
              aria-checked={isSelected}
              onClick={() => onChange(item.value)}
            >
              {isSelected && <Check size={14} aria-hidden="true" />}
              {item.label}
            </button>
          );
        })}
      </div>
      <p className="processing-mode-description">
        <strong>{selected.label}:</strong> {selected.description}
      </p>
    </div>
  );
}
