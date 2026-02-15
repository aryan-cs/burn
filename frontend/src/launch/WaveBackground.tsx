import { useEffect, useRef } from 'react'
import * as THREE from 'three'

const COLS = 300
const ROWS = 300
const POINT_COUNT = COLS * ROWS
const SPACING = 0.45

const DEFAULT_POINT_SIZE = 2.0
const DEFAULT_COLOR_R = 0.7
const DEFAULT_COLOR_G = 0.7
const DEFAULT_COLOR_B = 0.7
const DEFAULT_OPACITY = 0.35
const DEFAULT_AMPLITUDE = 0.8
const DEFAULT_SPEED = 0.4

function readCssNumber(style: CSSStyleDeclaration, name: string, fallback: number): number {
  const raw = style.getPropertyValue(name).trim()
  if (!raw) return fallback
  const parsed = Number.parseFloat(raw)
  return Number.isFinite(parsed) ? parsed : fallback
}

interface WaveBackgroundProps {
  className?: string
}

export default function WaveBackground({ className = 'wave-background' }: WaveBackgroundProps) {
  const mountRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const mountEl = mountRef.current
    if (!mountEl) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100)
    camera.position.set(0, 12, 14)
    camera.lookAt(0, 0, 0)

    const renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(1, 1, false)
    renderer.setClearColor(0x000000, 0)
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'
    renderer.domElement.style.pointerEvents = 'none'
    mountEl.appendChild(renderer.domElement)

    const geometry = new THREE.BufferGeometry()
    const positions = new Float32Array(POINT_COUNT * 3)
    const colors = new Float32Array(POINT_COUNT * 3)
    const baseX = new Float32Array(POINT_COUNT)
    const baseZ = new Float32Array(POINT_COUNT)

    const halfWidth = ((COLS - 1) * SPACING) * 0.5
    const halfDepth = ((ROWS - 1) * SPACING) * 0.5

    const initialColorR = DEFAULT_COLOR_R
    const initialColorG = DEFAULT_COLOR_G
    const initialColorB = DEFAULT_COLOR_B

    for (let row = 0; row < ROWS; row += 1) {
      const z = row * SPACING - halfDepth
      const rowOffset = row * COLS
      for (let col = 0; col < COLS; col += 1) {
        const i = rowOffset + col
        const x = col * SPACING - halfWidth
        const index3 = i * 3

        baseX[i] = x
        baseZ[i] = z

        positions[index3] = x
        positions[index3 + 1] = 0
        positions[index3 + 2] = z

        colors[index3] = initialColorR
        colors[index3 + 1] = initialColorG
        colors[index3 + 2] = initialColorB
      }
    }

    const positionAttribute = new THREE.BufferAttribute(positions, 3)
    const colorAttribute = new THREE.BufferAttribute(colors, 3)
    geometry.setAttribute('position', positionAttribute)
    geometry.setAttribute('color', colorAttribute)

    const material = new THREE.PointsMaterial({
      transparent: true,
      vertexColors: true,
      depthWrite: false,
      size: DEFAULT_POINT_SIZE * 0.04,
      opacity: DEFAULT_OPACITY,
    })

    const points = new THREE.Points(geometry, material)
    scene.add(points)

    const ax = -halfWidth * 0.55
    const az = -halfDepth * 0.35
    const bx = halfWidth * 0.5
    const bz = halfDepth * 0.45

    let lastColorR = initialColorR
    let lastColorG = initialColorG
    let lastColorB = initialColorB
    let frameId = 0
    let cssPollId = 0

    let pointSize = DEFAULT_POINT_SIZE
    let colorR = DEFAULT_COLOR_R
    let colorG = DEFAULT_COLOR_G
    let colorB = DEFAULT_COLOR_B
    let opacity = DEFAULT_OPACITY
    let amplitude = DEFAULT_AMPLITUDE
    let speed = DEFAULT_SPEED
    let resizeObserver: ResizeObserver | null = null

    const readCssTokens = () => {
      const css = window.getComputedStyle(mountEl)
      pointSize = readCssNumber(css, '--wave-point-size', DEFAULT_POINT_SIZE)
      colorR = readCssNumber(css, '--wave-color-r', DEFAULT_COLOR_R)
      colorG = readCssNumber(css, '--wave-color-g', DEFAULT_COLOR_G)
      colorB = readCssNumber(css, '--wave-color-b', DEFAULT_COLOR_B)
      opacity = readCssNumber(css, '--wave-opacity', DEFAULT_OPACITY)
      amplitude = readCssNumber(css, '--wave-amplitude', DEFAULT_AMPLITUDE)
      speed = readCssNumber(css, '--wave-speed', DEFAULT_SPEED)
    }

    const getMountSize = () => {
      const rect = mountEl.getBoundingClientRect()
      const width = Math.max(1, Math.floor(rect.width || window.innerWidth))
      const height = Math.max(1, Math.floor(rect.height || window.innerHeight))
      return { width, height }
    }

    const resize = () => {
      const { width, height } = getMountSize()
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
      renderer.setSize(width, height, false)
    }

    const animate = (now: number) => {
      const t = now * 0.001

      material.size = pointSize * 0.04
      material.opacity = opacity

      if (colorR !== lastColorR || colorG !== lastColorG || colorB !== lastColorB) {
        const colorArray = colorAttribute.array as Float32Array
        for (let i = 0; i < POINT_COUNT; i += 1) {
          const index3 = i * 3
          colorArray[index3] = colorR
          colorArray[index3 + 1] = colorG
          colorArray[index3 + 2] = colorB
        }
        colorAttribute.needsUpdate = true
        lastColorR = colorR
        lastColorG = colorG
        lastColorB = colorB
      }

      const positionArray = positionAttribute.array as Float32Array
      for (let i = 0; i < POINT_COUNT; i += 1) {
        const x = baseX[i]
        const z = baseZ[i]

        const dxA = x - ax
        const dzA = z - az
        const distanceA = Math.sqrt(dxA * dxA + dzA * dzA)
        const waveA = Math.sin(distanceA * 0.35 - t * speed)

        const dxB = x - bx
        const dzB = z - bz
        const distanceB = Math.sqrt(dxB * dxB + dzB * dzB)
        const waveB = Math.sin(distanceB * 0.25 + t * speed * 1.2)

        const waveC = Math.sin(x * 0.4 + t * speed)

        positionArray[i * 3 + 1] = (waveA + waveB + waveC) * amplitude
      }

      positionAttribute.needsUpdate = true
      renderer.render(scene, camera)
      frameId = window.requestAnimationFrame(animate)
    }

    window.addEventListener('resize', resize)
    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(() => {
        resize()
      })
      resizeObserver.observe(mountEl)
    }
    readCssTokens()
    resize()
    cssPollId = window.setInterval(readCssTokens, 200)
    frameId = window.requestAnimationFrame(animate)

    return () => {
      window.removeEventListener('resize', resize)
      if (resizeObserver) {
        resizeObserver.disconnect()
      }
      window.cancelAnimationFrame(frameId)
      window.clearInterval(cssPollId)
      geometry.dispose()
      material.dispose()
      renderer.dispose()
      if (renderer.domElement.parentElement === mountEl) {
        mountEl.removeChild(renderer.domElement)
      }
    }
  }, [])

  return <div ref={mountRef} className={className} />
}
