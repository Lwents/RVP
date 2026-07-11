import React, { useRef, PointerEvent, useEffect } from "react";
import { Check } from "lucide-react";
import { toAbsoluteApiUrl } from "../lib/api";

interface CustomBlurBox {
  x_percent: number;
  y_percent: number;
  width_percent: number;
  height_percent: number;
}

interface LivePreviewProps {
  subtitle_x_percent: number;
  subtitle_y_percent: number;
  subtitle_font_size: number;
  subtitle_box_enabled: boolean;
  subtitle_box_opacity: number;
  subtitle_box_height_percent: number;
  hard_subtitles: boolean;

  previewVideoUrl: string | null;

  logo_enabled: boolean;
  watermark_file_name: string | null;
  logo_x_percent: number;
  logo_y_percent: number;
  logo_width: number;

  blur_box_enabled: boolean;
  blur_box_y_percent: number;
  blur_box_height_percent: number;
  custom_blur_boxes: CustomBlurBox[];

  cinematic_bars_enabled: boolean;
  cinematic_bars_height_percent: number;

  setField: (key: any, value: any) => void;
  videoRef?: React.Ref<HTMLVideoElement>;
}

interface DragState {
  startX: number;
  startY: number;
  startValX: number;
  startValY: number;
  rect: DOMRect;
  type: "subtitle" | "logo" | "blur-box" | "custom-blur-box" | "custom-blur-resize";
  index?: number;
  boxWidth?: number;
  boxHeight?: number;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

export const LivePreview = React.memo(function LivePreview({
  subtitle_x_percent,
  subtitle_y_percent,
  subtitle_font_size,
  subtitle_box_enabled,
  subtitle_box_opacity,
  subtitle_box_height_percent,
  hard_subtitles,
  previewVideoUrl,
  logo_enabled,
  watermark_file_name,
  logo_x_percent,
  logo_y_percent,
  logo_width,
  blur_box_enabled,
  blur_box_y_percent,
  blur_box_height_percent,
  custom_blur_boxes,
  cinematic_bars_enabled,
  cinematic_bars_height_percent,
  setField,
  videoRef,
}: LivePreviewProps) {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const dragStartRef = useRef<DragState | null>(null);

  // References to draggable elements to apply high performance direct style mutations
  const subtitleNodeRef = useRef<HTMLDivElement | null>(null);
  const logoNodeRef = useRef<HTMLDivElement | null>(null);
  const blurBoxNodeRef = useRef<HTMLDivElement | null>(null);
  const customBoxNodeRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});

  const boxTop = clamp(subtitle_y_percent - subtitle_box_height_percent / 2, 0, 100 - subtitle_box_height_percent);

  // Reset positions if props change outside (e.g. from sliders)
  useEffect(() => {
    if (subtitleNodeRef.current && !dragStartRef.current) {
      subtitleNodeRef.current.style.left = `${subtitle_x_percent}%`;
      subtitleNodeRef.current.style.top = `${subtitle_y_percent}%`;
    }
  }, [subtitle_x_percent, subtitle_y_percent]);

  useEffect(() => {
    if (logoNodeRef.current && !dragStartRef.current) {
      logoNodeRef.current.style.left = `${logo_x_percent}%`;
      logoNodeRef.current.style.top = `${logo_y_percent}%`;
      logoNodeRef.current.style.transform = `translate(-${logo_x_percent}%, -${logo_y_percent}%)`;
    }
  }, [logo_x_percent, logo_y_percent]);

  useEffect(() => {
    if (blurBoxNodeRef.current && !dragStartRef.current) {
      const topVal = clamp(blur_box_y_percent, 0, 100 - blur_box_height_percent);
      blurBoxNodeRef.current.style.top = `${topVal}%`;
    }
  }, [blur_box_y_percent, blur_box_height_percent]);

  useEffect(() => {
    custom_blur_boxes.forEach((box, index) => {
      const el = customBoxNodeRefs.current[index];
      if (el && !dragStartRef.current) {
        el.style.left = `${box.x_percent}%`;
        el.style.top = `${box.y_percent}%`;
        el.style.width = `${box.width_percent}%`;
        el.style.height = `${box.height_percent}%`;
      }
    });
  }, [custom_blur_boxes]);

  // Pointer Handlers: Subtitle
  const beginSubtitleDrag = (e: PointerEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    const frame = frameRef.current;
    if (!frame) return;
    dragStartRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startValX: subtitle_x_percent,
      startValY: subtitle_y_percent,
      rect: frame.getBoundingClientRect(),
      type: "subtitle",
    };
  };

  const handleSubtitleMove = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragStartRef.current;
    if (!drag || drag.type !== "subtitle") return;
    const deltaX = e.clientX - drag.startX;
    const deltaY = e.clientY - drag.startY;
    const newX = clamp(drag.startValX + (deltaX / drag.rect.width) * 100, 5, 95);
    const newY = clamp(drag.startValY + (deltaY / drag.rect.height) * 100, 8, 94);

    e.currentTarget.style.left = `${newX}%`;
    e.currentTarget.style.top = `${newY}%`;
  };

  const handleSubtitleUp = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragStartRef.current;
    if (!drag || drag.type !== "subtitle") return;
    const deltaX = e.clientX - drag.startX;
    const deltaY = e.clientY - drag.startY;
    const newX = Math.round(clamp(drag.startValX + (deltaX / drag.rect.width) * 100, 5, 95));
    const newY = Math.round(clamp(drag.startValY + (deltaY / drag.rect.height) * 100, 8, 94));

    e.currentTarget.releasePointerCapture(e.pointerId);
    dragStartRef.current = null;
    setField("subtitle_x_percent", newX);
    setField("subtitle_y_percent", newY);
  };

  // Pointer Handlers: Logo Watermark
  const beginLogoDrag = (e: PointerEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    const frame = frameRef.current;
    if (!frame) return;
    dragStartRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startValX: logo_x_percent,
      startValY: logo_y_percent,
      rect: frame.getBoundingClientRect(),
      type: "logo",
    };
  };

  const handleLogoMove = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragStartRef.current;
    if (!drag || drag.type !== "logo") return;
    const deltaX = e.clientX - drag.startX;
    const deltaY = e.clientY - drag.startY;
    const newX = clamp(drag.startValX + (deltaX / drag.rect.width) * 100, 0, 100);
    const newY = clamp(drag.startValY + (deltaY / drag.rect.height) * 100, 0, 100);

    e.currentTarget.style.left = `${newX}%`;
    e.currentTarget.style.top = `${newY}%`;
    e.currentTarget.style.transform = `translate(-${newX}%, -${newY}%)`;
  };

  const handleLogoUp = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragStartRef.current;
    if (!drag || drag.type !== "logo") return;
    const deltaX = e.clientX - drag.startX;
    const deltaY = e.clientY - drag.startY;
    const newX = Math.round(clamp(drag.startValX + (deltaX / drag.rect.width) * 100, 0, 100));
    const newY = Math.round(clamp(drag.startValY + (deltaY / drag.rect.height) * 100, 0, 100));

    e.currentTarget.releasePointerCapture(e.pointerId);
    dragStartRef.current = null;
    setField("logo_x_percent", newX);
    setField("logo_y_percent", newY);
  };

  const removeLogo = () => {
    dragStartRef.current = null;
    setField("watermark_file_name", null);
    setField("logo_enabled", false);
  };

  // Pointer Handlers: original Subtitle Blur Box
  const beginBlurBoxDrag = (e: PointerEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    const frame = frameRef.current;
    if (!frame) return;
    dragStartRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startValX: 0,
      startValY: blur_box_y_percent,
      rect: frame.getBoundingClientRect(),
      type: "blur-box",
    };
  };

  const handleBlurBoxMove = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragStartRef.current;
    if (!drag || drag.type !== "blur-box") return;
    const deltaY = e.clientY - drag.startY;
    const newY = clamp(drag.startValY + (deltaY / drag.rect.height) * 100, 0, 100 - blur_box_height_percent);

    e.currentTarget.style.top = `${newY}%`;
  };

  const handleBlurBoxUp = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragStartRef.current;
    if (!drag || drag.type !== "blur-box") return;
    const deltaY = e.clientY - drag.startY;
    const newY = Math.round(clamp(drag.startValY + (deltaY / drag.rect.height) * 100, 0, 100 - blur_box_height_percent));

    e.currentTarget.releasePointerCapture(e.pointerId);
    dragStartRef.current = null;
    setField("blur_box_y_percent", newY);
  };

  // Pointer Handlers: Custom Blur Box Dragging
  const beginCustomBoxDrag = (e: PointerEvent<HTMLDivElement>, index: number) => {
    e.stopPropagation();
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    const frame = frameRef.current;
    if (!frame) return;
    const box = custom_blur_boxes[index];
    dragStartRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startValX: box.x_percent,
      startValY: box.y_percent,
      rect: frame.getBoundingClientRect(),
      type: "custom-blur-box",
      index,
    };
  };

  const handleCustomBoxMove = (e: PointerEvent<HTMLDivElement>, index: number) => {
    const drag = dragStartRef.current;
    if (!drag || drag.type !== "custom-blur-box" || drag.index !== index) return;
    const deltaX = e.clientX - drag.startX;
    const deltaY = e.clientY - drag.startY;
    const box = custom_blur_boxes[index];
    const newX = clamp(drag.startValX + (deltaX / drag.rect.width) * 100, 0, 100 - box.width_percent);
    const newY = clamp(drag.startValY + (deltaY / drag.rect.height) * 100, 0, 100 - box.height_percent);

    e.currentTarget.style.left = `${newX}%`;
    e.currentTarget.style.top = `${newY}%`;
  };

  const handleCustomBoxUp = (e: PointerEvent<HTMLDivElement>, index: number) => {
    const drag = dragStartRef.current;
    if (!drag || drag.type !== "custom-blur-box" || drag.index !== index) return;
    const deltaX = e.clientX - drag.startX;
    const deltaY = e.clientY - drag.startY;
    const box = custom_blur_boxes[index];
    const newX = Math.round(clamp(drag.startValX + (deltaX / drag.rect.width) * 100, 0, 100 - box.width_percent));
    const newY = Math.round(clamp(drag.startValY + (deltaY / drag.rect.height) * 100, 0, 100 - box.height_percent));

    e.currentTarget.releasePointerCapture(e.pointerId);
    dragStartRef.current = null;

    const boxes = [...custom_blur_boxes];
    boxes[index] = { ...boxes[index], x_percent: newX, y_percent: newY };
    setField("custom_blur_boxes", boxes);
  };

  // Pointer Handlers: Custom Blur Box Resizing
  const beginCustomBoxResize = (e: PointerEvent<HTMLDivElement>, index: number) => {
    e.stopPropagation();
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    const frame = frameRef.current;
    if (!frame) return;
    const box = custom_blur_boxes[index];
    dragStartRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startValX: box.x_percent,
      startValY: box.y_percent,
      boxWidth: box.width_percent,
      boxHeight: box.height_percent,
      rect: frame.getBoundingClientRect(),
      type: "custom-blur-resize",
      index,
    };
  };

  const handleCustomBoxResizeMove = (e: PointerEvent<HTMLDivElement>, index: number) => {
    const drag = dragStartRef.current;
    if (!drag || drag.type !== "custom-blur-resize" || drag.index !== index) return;
    const deltaX = e.clientX - drag.startX;
    const deltaY = e.clientY - drag.startY;

    const newWidth = clamp((drag.boxWidth || 0) + (deltaX / drag.rect.width) * 100, 2, 100 - drag.startValX);
    const newHeight = clamp((drag.boxHeight || 0) + (deltaY / drag.rect.height) * 100, 2, 100 - drag.startValY);

    const parent = e.currentTarget.parentElement;
    if (parent) {
      parent.style.width = `${newWidth}%`;
      parent.style.height = `${newHeight}%`;
    }
  };

  const handleCustomBoxResizeUp = (e: PointerEvent<HTMLDivElement>, index: number) => {
    const drag = dragStartRef.current;
    if (!drag || drag.type !== "custom-blur-resize" || drag.index !== index) return;
    const deltaX = e.clientX - drag.startX;
    const deltaY = e.clientY - drag.startY;

    const newWidth = Math.round(clamp((drag.boxWidth || 0) + (deltaX / drag.rect.width) * 100, 2, 100 - drag.startValX));
    const newHeight = Math.round(clamp((drag.boxHeight || 0) + (deltaY / drag.rect.height) * 100, 2, 100 - drag.startValY));

    e.currentTarget.releasePointerCapture(e.pointerId);
    dragStartRef.current = null;

    const boxes = [...custom_blur_boxes];
    boxes[index] = {
      ...boxes[index],
      width_percent: newWidth,
      height_percent: newHeight,
    };
    setField("custom_blur_boxes", boxes);
  };

  return (
    <div className="live-editor">
      <div className="video-frame" ref={frameRef}>
        {previewVideoUrl ? (
          <video
            ref={videoRef}
            className="preview-video-element"
            src={previewVideoUrl}
            controls
            autoPlay
            loop
            muted
            playsInline
            onError={(e) => {
              (e.target as HTMLVideoElement).style.display = "none";
            }}
            onLoadStart={(e) => {
              (e.target as HTMLVideoElement).style.display = "block";
            }}
          />
        ) : (
          <div className="sample-scene">
            <div className="scene-grid" />
            <div className="scene-person" />
            <div className="scene-caption">Preview frame</div>
          </div>
        )}

        {cinematic_bars_enabled && cinematic_bars_height_percent > 0 && (
          <>
            <div className="cinematic-bar top" style={{ height: `${cinematic_bars_height_percent}%` }} />
            <div className="cinematic-bar bottom" style={{ height: `${cinematic_bars_height_percent}%` }} />
          </>
        )}

        {blur_box_enabled && (
          <div
            ref={blurBoxNodeRef}
            className="blur-box-layer"
            style={{
              top: `${clamp(blur_box_y_percent, 0, 100 - blur_box_height_percent)}%`,
              height: `${blur_box_height_percent}%`,
            }}
            onPointerDown={beginBlurBoxDrag}
            onPointerMove={handleBlurBoxMove}
            onPointerUp={handleBlurBoxUp}
            onPointerCancel={handleBlurBoxUp}
          />
        )}

        {custom_blur_boxes?.map((box, index) => (
          <div
            key={index}
            ref={(el) => {
              customBoxNodeRefs.current[index] = el;
            }}
            className="custom-blur-box-layer"
            style={{
              left: `${box.x_percent}%`,
              top: `${box.y_percent}%`,
              width: `${box.width_percent}%`,
              height: `${box.height_percent}%`,
            }}
            onPointerDown={(e) => beginCustomBoxDrag(e, index)}
            onPointerMove={(e) => handleCustomBoxMove(e, index)}
            onPointerUp={(e) => handleCustomBoxUp(e, index)}
            onPointerCancel={(e) => handleCustomBoxUp(e, index)}
            onDoubleClick={() => {
              const boxes = [...custom_blur_boxes];
              boxes.splice(index, 1);
              setField("custom_blur_boxes", boxes);
            }}
          >
            <button
              type="button"
              className="custom-blur-delete-btn"
              onClick={(e) => {
                e.stopPropagation();
                const boxes = [...custom_blur_boxes];
                boxes.splice(index, 1);
                setField("custom_blur_boxes", boxes);
              }}
              title="Xoá vùng làm mờ"
            >
              ×
            </button>
            <div
              className="resize-handle"
              onPointerDown={(e) => beginCustomBoxResize(e, index)}
              onPointerMove={(e) => handleCustomBoxResizeMove(e, index)}
              onPointerUp={(e) => handleCustomBoxResizeUp(e, index)}
              onPointerCancel={(e) => handleCustomBoxResizeUp(e, index)}
            />
          </div>
        ))}

        {logo_enabled && watermark_file_name && (
          <div
            ref={logoNodeRef}
            className="watermark-logo-wrapper"
            style={{
              left: `${logo_x_percent}%`,
              top: `${logo_y_percent}%`,
              transform: `translate(-${logo_x_percent}%, -${logo_y_percent}%)`,
            }}
            onPointerDown={beginLogoDrag}
            onPointerMove={handleLogoMove}
            onPointerUp={handleLogoUp}
            onPointerCancel={handleLogoUp}
          >
            <button
              type="button"
              className="custom-blur-delete-btn watermark-delete-btn"
              onClick={(e) => {
                e.stopPropagation();
                removeLogo();
              }}
              title="Xoa logo"
            >
              ×
            </button>
            <img
              src={toAbsoluteApiUrl(`/api/uploads/watermark/${watermark_file_name}`)}
              className="watermark-logo-layer"
              style={{
                width: `${Math.max(32, logo_width * 0.6)}px`,
              }}
              alt="Logo"
              draggable={false}
            />
          </div>
        )}

        {hard_subtitles && subtitle_box_enabled && (
          <div
            className="blur-band"
            style={{
              top: `${boxTop}%`,
              height: `${subtitle_box_height_percent}%`,
              opacity: subtitle_box_opacity / 100,
            }}
          />
        )}

        {hard_subtitles && (
          <div
            ref={subtitleNodeRef}
            className="subtitle-layer"
            style={{
              left: `${subtitle_x_percent}%`,
              top: `${subtitle_y_percent}%`,
              fontSize: `${Math.max(16, subtitle_font_size * 0.72)}px`,
            }}
            onPointerDown={beginSubtitleDrag}
            onPointerMove={handleSubtitleMove}
            onPointerUp={handleSubtitleUp}
            onPointerCancel={handleSubtitleUp}
          >
            <span>Phụ đề sẽ nằm ở đây sau khi render</span>
          </div>
        )}
      </div>
    </div>
  );
}, (prev, next) => {
  // Memoization comparative fields to skip re-renders if unrelated states change
  return (
    prev.subtitle_x_percent === next.subtitle_x_percent &&
    prev.subtitle_y_percent === next.subtitle_y_percent &&
    prev.subtitle_font_size === next.subtitle_font_size &&
    prev.subtitle_box_enabled === next.subtitle_box_enabled &&
    prev.subtitle_box_opacity === next.subtitle_box_opacity &&
    prev.subtitle_box_height_percent === next.subtitle_box_height_percent &&
    prev.hard_subtitles === next.hard_subtitles &&
    prev.previewVideoUrl === next.previewVideoUrl &&
    prev.logo_enabled === next.logo_enabled &&
    prev.watermark_file_name === next.watermark_file_name &&
    prev.logo_x_percent === next.logo_x_percent &&
    prev.logo_y_percent === next.logo_y_percent &&
    prev.logo_width === next.logo_width &&
    prev.blur_box_enabled === next.blur_box_enabled &&
    prev.blur_box_y_percent === next.blur_box_y_percent &&
    prev.blur_box_height_percent === next.blur_box_height_percent &&
    prev.cinematic_bars_enabled === next.cinematic_bars_enabled &&
    prev.cinematic_bars_height_percent === next.cinematic_bars_height_percent &&
    prev.videoRef === next.videoRef &&
    prev.custom_blur_boxes === next.custom_blur_boxes // reference check is sufficient since we replace array reference on change
  );
});
