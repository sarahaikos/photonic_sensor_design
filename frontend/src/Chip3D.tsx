import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

export type Chip3DDevice =
  | {
      id: string;
      type: "ring";
      x: number;
      y: number;
      radius_um: number;
      config: "all-pass" | "add-drop";
    }
  | { id: string; type: "waveguide"; x: number; y: number; length_um: number }
  | { id: string; type: "coupler"; x: number; y: number; gap_nm: number; length_um: number }
  | {
      id: string;
      type: "detector";
      x: number;
      y: number;
      optical_power_uw: number;
      responsivity_a_per_w: number;
      dark_current_na: number;
      bandwidth_mhz: number;
      load_ohm: number;
    };

type Props = {
  devices: Chip3DDevice[];
  selectedId: string | null;
  widthNm: number;
  heightNm: number;
  nClad: number;
  onSelect: (id: string | null) => void;
  onPlaceSensor: () => void;
};

const ACCENT = 0x0f766e;
const SI = 0x5c6770;
const THROUGH = 0x2f6f6a;
const DROP = 0x1d6a93;
const GE = 0xb4bcc6;
const BOX = 0xe8e4d8;

type BusRole = "through" | "drop" | "waveguide";

type Engine = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  labelRenderer: CSS2DRenderer;
  controls: OrbitControls;
  root: THREE.Group;
  pickables: THREE.Object3D[];
  baseColors: Map<THREE.Mesh, number>;
  raf: number;
  ro: ResizeObserver;
  layoutKey: string;
};

function wgHalf(lengthUm: number) {
  return Math.max(50, Math.min(150, lengthUm * 0.7)) / 2;
}

function ringPx(radiusUm: number) {
  return Math.max(18, Math.min(34, radiusUm * 2.2));
}

function nearestRing(x: number, y: number, devices: Chip3DDevice[]) {
  const rings = devices.filter((d): d is Extract<Chip3DDevice, { type: "ring" }> => d.type === "ring");
  if (!rings.length) return null;
  return rings.reduce((best, r) => {
    const dBest = (best.x - x) ** 2 + (best.y - y) ** 2;
    const dR = (r.x - x) ** 2 + (r.y - y) ** 2;
    return dR < dBest ? r : best;
  });
}

function waveguides(devices: Chip3DDevice[]) {
  return devices.filter((d): d is Extract<Chip3DDevice, { type: "waveguide" }> => d.type === "waveguide");
}

function busRole(wg: Extract<Chip3DDevice, { type: "waveguide" }>, devices: Chip3DDevice[]): BusRole {
  const buses = waveguides(devices);
  const ring = devices.find(
    (d): d is Extract<Chip3DDevice, { type: "ring" }> => d.type === "ring" && d.config === "add-drop"
  );
  if (ring && buses.length >= 2) return wg.y < ring.y ? "drop" : "through";
  return "waveguide";
}

function couplerRole(c: Extract<Chip3DDevice, { type: "coupler" }>, devices: Chip3DDevice[]): BusRole {
  const ring = devices.find(
    (d): d is Extract<Chip3DDevice, { type: "ring" }> => d.type === "ring" && d.config === "add-drop"
  );
  if (!ring) return "waveguide";
  return c.y < ring.y ? "drop" : "through";
}

function roleColor(role: BusRole) {
  if (role === "through") return THROUGH;
  if (role === "drop") return DROP;
  return SI;
}

function layoutKeyOf(devices: Chip3DDevice[], widthNm: number, heightNm: number, nClad: number) {
  return JSON.stringify({
    widthNm,
    heightNm,
    nClad,
    devices: devices.map((d) => {
      if (d.type === "ring") return [d.id, d.type, d.x, d.y, d.radius_um, d.config];
      if (d.type === "waveguide") return [d.id, d.type, d.x, d.y, d.length_um];
      if (d.type === "coupler") return [d.id, d.type, d.x, d.y, d.gap_nm, d.length_um];
      return [d.id, d.type, d.x, d.y];
    }),
  });
}

function makeMat(color: number) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.62,
    metalness: 0.04,
    emissive: 0x000000,
    emissiveIntensity: 0,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
}

function makeLabel(text: string, tone: "through" | "drop" | "ring" | "muted" = "muted") {
  const el = document.createElement("div");
  el.className = `iso-float-label iso-float-${tone}`;
  el.textContent = text;
  return new CSS2DObject(el);
}

function clearGroup(root: THREE.Group) {
  while (root.children.length) {
    const obj = root.children[0];
    root.remove(obj);
    obj.traverse((child) => {
      if (child instanceof CSS2DObject) {
        child.element.remove();
      }
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        const m = child.material;
        if (Array.isArray(m)) m.forEach((x) => x.dispose());
        else m.dispose();
      }
    });
  }
}

function applySelection(engine: Engine, selectedId: string | null) {
  for (const obj of engine.pickables) {
    if (!(obj instanceof THREE.Mesh)) continue;
    const material = obj.material;
    if (!(material instanceof THREE.MeshStandardMaterial)) continue;
    const base = engine.baseColors.get(obj) ?? SI;
    const selected = obj.userData.deviceId === selectedId;
    material.color.setHex(selected ? ACCENT : base);
    material.emissive.setHex(selected ? ACCENT : 0x000000);
    material.emissiveIntensity = selected ? 0.22 : 0;
  }
}

function buildLayout(
  engine: Engine,
  devices: Chip3DDevice[],
  widthNm: number,
  heightNm: number,
  nClad: number,
  selectedId: string | null,
  resetCamera: boolean
) {
  clearGroup(engine.root);
  engine.pickables.length = 0;
  engine.baseColors.clear();

  const boxH = 22;
  const siH = Math.max(12, Math.min(44, heightNm * 0.13));
  const coreW = Math.max(7, Math.min(14, widthNm * 0.015));
  const cladName = nClad === 1 ? "air" : nClad > 1.4 ? "oxide" : "water";
  const addDrop = devices.some((d) => d.type === "ring" && d.config === "add-drop");

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const d of devices) {
    if (d.type === "waveguide") {
      const half = wgHalf(d.length_um);
      minX = Math.min(minX, d.x - half);
      maxX = Math.max(maxX, d.x + half);
      minY = Math.min(minY, d.y - coreW);
      maxY = Math.max(maxY, d.y + coreW);
    } else if (d.type === "coupler") {
      const half = Math.max(28, Math.min(50, d.length_um * 3));
      const gap = Math.max(8, Math.min(16, d.gap_nm * 0.04));
      minX = Math.min(minX, d.x - half);
      maxX = Math.max(maxX, d.x + half);
      minY = Math.min(minY, d.y - gap - coreW);
      maxY = Math.max(maxY, d.y + gap + coreW);
    } else if (d.type === "ring") {
      const r = ringPx(d.radius_um);
      minX = Math.min(minX, d.x - r);
      maxX = Math.max(maxX, d.x + r);
      minY = Math.min(minY, d.y - r);
      maxY = Math.max(maxY, d.y + r);
    } else {
      minX = Math.min(minX, d.x - 16);
      maxX = Math.max(maxX, d.x + 16);
      minY = Math.min(minY, d.y - 14);
      maxY = Math.max(maxY, d.y + 14);
    }
  }

  const margin = 44;
  const chipX = minX - margin;
  const chipY = minY - margin;
  const chipW = maxX - minX + margin * 2;
  const chipD = maxY - minY + margin * 2;
  const cx = chipX + chipW / 2;
  const cy = chipY + chipD / 2;

  const toWorld = (x: number, y: number) => ({ x: x - cx, z: y - cy });

  const addBox = (
    x: number,
    y: number,
    w: number,
    d: number,
    z0: number,
    h: number,
    color: number,
    deviceId?: string
  ) => {
    const geo = new THREE.BoxGeometry(w, h, d);
    const material = makeMat(color);
    const mesh = new THREE.Mesh(geo, material);
    const p = toWorld(x + w / 2, y + d / 2);
    mesh.position.set(p.x, z0 + h / 2, p.z);
    if (deviceId) {
      mesh.userData.deviceId = deviceId;
      engine.pickables.push(mesh);
      engine.baseColors.set(mesh, color);
    }
    engine.root.add(mesh);
    return mesh;
  };

  addBox(chipX, chipY, chipW, chipD, -boxH, boxH, BOX);

  for (const d of devices) {
    if (d.type === "waveguide") {
      const half = wgHalf(d.length_um);
      const role = busRole(d, devices);
      const color = roleColor(role);
      addBox(d.x - half, d.y - coreW / 2, half * 2, coreW, 0, siH, color, d.id);
      const p = toWorld(d.x - half + 10, d.y);
      const label = makeLabel(role === "waveguide" ? "bus" : role, role === "waveguide" ? "muted" : role);
      label.position.set(p.x, siH + 16, p.z);
      engine.root.add(label);
    } else if (d.type === "coupler") {
      // Coupler is already implied by bus + ring proximity; draw a short bridge only
      // so it does not z-fight with the full bus extrusion.
      const half = Math.max(18, Math.min(34, d.length_um * 2.2));
      const gap = Math.max(8, Math.min(16, d.gap_nm * 0.04));
      const ringNear = nearestRing(d.x, d.y, devices);
      const gapDir = ringNear && ringNear.y > d.y ? gap : -gap;
      const role = couplerRole(d, devices);
      const color = roleColor(role);
      const bridgeY = Math.min(d.y, d.y + gapDir);
      const bridgeD = Math.abs(gapDir) + coreW;
      addBox(d.x - half, bridgeY - coreW / 2, half * 2, bridgeD, siH * 0.15, siH * 0.55, color, d.id);
      if (addDrop) {
        const p = toWorld(d.x, d.y + gapDir * 0.5);
        const label = makeLabel(role === "drop" ? "drop κ" : "through κ", role === "waveguide" ? "muted" : role);
        label.position.set(p.x, siH + 20, p.z);
        engine.root.add(label);
      }
    } else if (d.type === "ring") {
      const r = ringPx(d.radius_um);
      const tube = Math.max(2.4, coreW * 0.48);
      const geo = new THREE.TorusGeometry(Math.max(tube + 2, r - tube * 0.15), tube, 20, 72);
      const material = makeMat(SI);
      const mesh = new THREE.Mesh(geo, material);
      const p = toWorld(d.x, d.y);
      mesh.position.set(p.x, siH / 2, p.z);
      mesh.rotation.x = Math.PI / 2;
      mesh.userData.deviceId = d.id;
      engine.pickables.push(mesh);
      engine.baseColors.set(mesh, SI);
      engine.root.add(mesh);
      const label = makeLabel(d.config === "add-drop" ? "ring (add-drop)" : "ring", "ring");
      label.position.set(p.x, siH + 24, p.z);
      engine.root.add(label);
    } else {
      addBox(d.x - 12, d.y - 9, 24, 18, 0, siH * 0.75, GE, d.id);
      const p = toWorld(d.x, d.y);
      const label = makeLabel("detector", "muted");
      label.position.set(p.x, siH + 16, p.z);
      engine.root.add(label);
    }
  }

  applySelection(engine, selectedId);

  const span = Math.max(chipW, chipD, 160);
  if (resetCamera) {
    engine.camera.position.set(span * 0.62, span * 0.52, span * 0.78);
    engine.controls.target.set(0, siH * 0.35, 0);
    engine.controls.update();
  }

  return { addDrop, cladName };
}

export function Chip3D({
  devices,
  selectedId,
  widthNm,
  heightNm,
  nClad,
  onSelect,
  onPlaceSensor,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const labelRef = useRef<HTMLParagraphElement>(null);
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const engineRef = useRef<Engine | null>(null);
  const selectedRef = useRef(selectedId);
  selectedRef.current = selectedId;

  const layoutKey = useMemo(
    () => layoutKeyOf(devices, widthNm, heightNm, nClad),
    [devices, widthNm, heightNm, nClad]
  );

  // One WebGL context for the life of the host.
  useEffect(() => {
    const host = hostRef.current;
    if (!host || devices.length === 0) return;

    host.querySelectorAll("canvas, .iso-label-layer").forEach((node) => node.remove());

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf3f3ef);

    const camera = new THREE.PerspectiveCamera(40, 1, 1, 5000);
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.style.display = "block";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    host.appendChild(renderer.domElement);

    const labelRenderer = new CSS2DRenderer();
    labelRenderer.domElement.className = "iso-label-layer";
    host.appendChild(labelRenderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.minDistance = 90;
    controls.maxDistance = 1000;
    controls.maxPolarAngle = Math.PI * 0.48;
    controls.rotateSpeed = 0.85;
    controls.zoomSpeed = 0.9;

    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const key = new THREE.DirectionalLight(0xffffff, 0.85);
    key.position.set(160, 220, 140);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xdfe8e6, 0.4);
    fill.position.set(-140, 100, -100);
    scene.add(fill);

    const root = new THREE.Group();
    scene.add(root);

    const engine: Engine = {
      scene,
      camera,
      renderer,
      labelRenderer,
      controls,
      root,
      pickables: [],
      baseColors: new Map(),
      raf: 0,
      ro: null as unknown as ResizeObserver,
      layoutKey: "",
    };
    engineRef.current = engine;

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let pointerDown = false;
    let dragged = false;
    let downX = 0;
    let downY = 0;

    function resize() {
      const el = hostRef.current;
      if (!el) return;
      const w = el.clientWidth || 480;
      const h = el.clientHeight || 360;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
      labelRenderer.setSize(w, h);
    }
    resize();
    const ro = new ResizeObserver(() => {
      resize();
    });
    ro.observe(host);
    engine.ro = ro;

    function onPointerDown(e: PointerEvent) {
      pointerDown = true;
      dragged = false;
      downX = e.clientX;
      downY = e.clientY;
    }
    function onPointerMove(e: PointerEvent) {
      if (!pointerDown) return;
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 5) dragged = true;
    }
    function onPointerUp(e: PointerEvent) {
      if (!pointerDown) return;
      pointerDown = false;
      if (dragged) return;
      const rect = renderer.domElement.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(engine.pickables, false);
      const id = hits[0]?.object.userData.deviceId as string | undefined;
      onSelectRef.current(id ?? null);
    }
    function onPointerLeave() {
      pointerDown = false;
    }

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("pointerleave", onPointerLeave);

    const tick = () => {
      engine.raf = requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };
    tick();

    return () => {
      cancelAnimationFrame(engine.raf);
      ro.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
      controls.dispose();
      clearGroup(root);
      renderer.dispose();
      renderer.domElement.remove();
      labelRenderer.domElement.remove();
      if (engineRef.current === engine) engineRef.current = null;
    };
    // Mount once while devices exist; layout updates happen in the effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [devices.length > 0]);

  // Rebuild meshes only when the layout actually changes.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || devices.length === 0) return;
    if (engine.layoutKey === layoutKey && engine.root.children.length > 0) return;
    const resetCamera = engine.layoutKey === "";
    const meta = buildLayout(engine, devices, widthNm, heightNm, nClad, selectedRef.current, resetCamera);
    engine.layoutKey = layoutKey;
    if (labelRef.current) {
      labelRef.current.textContent = meta.addDrop
        ? `BOX · Si ${widthNm}×${heightNm} nm · ${meta.cladName} · add-drop`
        : `BOX · Si ${widthNm}×${heightNm} nm · ${meta.cladName}`;
    }
  }, [layoutKey, devices, widthNm, heightNm, nClad]);

  // Selection highlight without tearing down the scene.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || engine.root.children.length === 0) return;
    applySelection(engine, selectedId);
  }, [selectedId]);

  if (devices.length === 0) {
    return (
      <div className="empty-board">
        <div className="empty-mark">
          <svg viewBox="0 0 88 56" aria-hidden="true">
            <line x1="6" y1="40" x2="82" y2="40" />
            <circle cx="44" cy="24" r="14" />
          </svg>
        </div>
        <p>Empty chip</p>
        <p className="muted">Place a part, then open 3D to see the SOI stack.</p>
        <button type="button" className="primary" onClick={onPlaceSensor}>
          Place a ring sensor
        </button>
      </div>
    );
  }

  return (
    <div className="board-stage iso-webgl-stage">
      <div className="iso-webgl-host" ref={hostRef}>
        <p className="iso-webgl-label" ref={labelRef} />
      </div>
      <p className="board-hint">
        Drag to orbit, scroll to zoom. Click a part to select it — layout look only, not a 3D FDTD
        mesh.
      </p>
    </div>
  );
}
