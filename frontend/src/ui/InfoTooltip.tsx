import { useRef, useState, useEffect, useCallback, useLayoutEffect } from 'react'
import { createPortal } from 'react-dom'

interface InfoTooltipProps {
  text: string
  /** Optional title shown at the top of the popup */
  title?: string
  /** Where the popup appears relative to the button */
  position?: 'top' | 'bottom' | 'left' | 'right'
}

export function InfoTooltip({ text, title, position = 'top' }: InfoTooltipProps) {
  const [open, setOpen] = useState(false)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const popupRef = useRef<HTMLDivElement>(null)
  const [coords, setCoords] = useState<{ top: number; left: number }>({ top: 0, left: 0 })

  const toggle = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    e.preventDefault()
    setOpen((v) => !v)
  }, [])

  // Position the popup relative to the trigger button
  useLayoutEffect(() => {
    if (!open || !buttonRef.current) return
    const rect = buttonRef.current.getBoundingClientRect()
    const popupWidth = 260
    const popupEstimatedHeight = 120

    let top = 0
    let left = 0

    switch (position) {
      case 'right':
        top = rect.top + rect.height / 2 - popupEstimatedHeight / 2
        left = rect.right + 10
        break
      case 'left':
        top = rect.top + rect.height / 2 - popupEstimatedHeight / 2
        left = rect.left - popupWidth - 10
        break
      case 'bottom':
        top = rect.bottom + 10
        left = rect.left + rect.width / 2 - popupWidth / 2
        break
      case 'top':
      default:
        top = rect.top - popupEstimatedHeight - 10
        left = rect.left + rect.width / 2 - popupWidth / 2
        break
    }

    // Clamp to viewport edges
    left = Math.max(8, Math.min(left, window.innerWidth - popupWidth - 8))
    top = Math.max(8, top)

    setCoords({ top, left })
  }, [open, position])

  // Close on click outside or Escape
  useEffect(() => {
    if (!open) return
    const handleClick = (e: MouseEvent) => {
      if (
        popupRef.current &&
        !popupRef.current.contains(e.target as Node) &&
        buttonRef.current &&
        !buttonRef.current.contains(e.target as Node)
      ) {
        setOpen(false)
      }
    }
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('mousedown', handleClick)
      document.removeEventListener('keydown', handleKey)
    }
  }, [open])

  return (
    <span className="info-tooltip-wrapper" onClick={(e) => e.stopPropagation()}>
      <button
        ref={buttonRef}
        type="button"
        className={`info-tooltip-trigger ${open ? 'info-tooltip-trigger-active' : ''}`}
        onClick={toggle}
        aria-label="More info"
        aria-expanded={open}
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 16 16"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
          <text
            x="8"
            y="12"
            textAnchor="middle"
            fill="currentColor"
            fontSize="10"
            fontWeight="600"
            fontFamily="inherit"
          >
            i
          </text>
        </svg>
      </button>
      {open &&
        createPortal(
          <div
            ref={popupRef}
            className="info-popup"
            role="dialog"
            aria-modal="false"
            style={{
              position: 'fixed',
              top: coords.top,
              left: coords.left,
            }}
          >
            <div className="info-popup-header">
              <span className="info-popup-title">{title ?? 'Info'}</span>
              <button
                type="button"
                className="info-popup-close"
                onClick={(e) => {
                  e.stopPropagation()
                  setOpen(false)
                }}
                aria-label="Close"
              >
                ✕
              </button>
            </div>
            <p className="info-popup-body">{text}</p>
          </div>,
          document.body,
        )}
    </span>
  )
}
