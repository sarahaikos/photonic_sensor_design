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
    }
  | {
      id: string;
      type: "heater";
      x: number;
      y: number;
      power_mw: number;
      width_um: number;
      length_um: number;
    };

type Props = {
  devices: Chip3DDevice[];
  selectedId: string | null;
  widthNm: number;
  heightNm: number;
  nClad: number;
  active?: boolean;
  onSelect: (id: string | null) => void;
  onPlaceSensor: () => void;
};

const ACCENT = 0x0f766e;
const SI = 0x5c6770;
const THROUGH = 0x2f6f6a;
const DROP = 0x1d6a93;
const GE = 0xb4bcc6;
const HEATER = 0xd4b06a;
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
  sharedMats: Map<number, THREE.MeshStandardMaterial>;
  selectedMat: THREE.MeshStandardMaterial;
  deviceCenters: Map<string, THREE.Vector3>;
  layoutKey: string;
  siH: number;
  span: number;
  invalidate: (frames?: number) => void;
  resize: () => void;
  dispose: () => void;
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
      if (d.type === "heater") return [d.id, d.type, d.x, d.y, d.power_mw, d.width_um, d.length_um];
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

function sharedMat(engine: Engine, color: number) {
  let mat = engine.sharedMats.get(color);
  if (!mat) {
    mat = makeMat(color);
    mat.userData.shared = true;
    engine.sharedMats.set(color, mat);
  }
  return mat;
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
        const mats = Array.isArray(m) ? m : [m];
        for (const mat of mats) {
          if (!mat.userData?.shared) mat.dispose();
        }
      }
    });
  }
}

function applySelection(engine: Engine, selectedId: string | null) {
  for (const obj of engine.pickables) {
    if (!(obj instanceof THREE.Mesh)) continue;
    const base = engine.baseColors.get(obj) ?? SI;
    const selected = obj.userData.deviceId === selectedId;
    obj.material = selected ? engine.selectedMat : sharedMat(engine, base);
  }
  engine.invalidate(2);
}

function resetCamera(engine: Engine) {
  const span = engine.span || 200;
  engine.camera.position.set(span * 0.62, span * 0.52, span * 0.78);
  engine.controls.target.set(0, engine.siH * 0.35, 0);
  engine.controls.update();
  engine.invalidate(8);
}

function frameSelection(engine: Engine, selectedId: string | null) {
  if (!selectedId) return;
  const center = engine.deviceCenters.get(selectedId);
  if (!center) return;
  engine.controls.target.copy(center);
  const offset = engine.camera.position.clone().sub(engine.controls.target);
  const dist = Math.max(120, Math.min(engine.span * 0.85, offset.length()));
  offset.setLength(dist);
  engine.camera.position.copy(center).add(offset);
  engine.controls.update();
  engine.invalidate(8);
}

function buildLayout(
  engine: Engine,
  devices: Chip3DDevice[],
  widthNm: number,
  heightNm: number,
  nClad: number,
  selectedId: string | null,
  doResetCamera: boolean
) {
  clearGroup(engine.root);
  engine.pickables.length = 0;
  engine.baseColors.clear();
  engine.deviceCenters.clear();

  const boxH = 22;
  const siH = Math.max(12, Math.min(44, heightNm * 0.13));
  const coreW = Math.max(7, Math.min(14, widthNm * 0.015));
  const cladName = nClad === 1 ? "air" : nClad > 1.4 ? "oxide" : "water";
  const addDrop = devices.some((d) => d.type === "ring" && d.config === "add-drop");
  engine.siH = siH;

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
    } else if (d.type === "heater") {
      minX = Math.min(minX, d.x - 18);
      maxX = Math.max(maxX, d.x + 18);
      minY = Math.min(minY, d.y - 8);
      maxY = Math.max(maxY, d.y + 8);
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
    const mesh = new THREE.Mesh(geo, sharedMat(engine, color));
    const p = toWorld(x + w / 2, y + d / 2);
    mesh.position.set(p.x, z0 + h / 2, p.z);
    if (deviceId) {
      mesh.userData.deviceId = deviceId;
      engine.pickables.push(mesh);
      engine.baseColors.set(mesh, color);
      engine.deviceCenters.set(deviceId, new THREE.Vector3(p.x, z0 + h / 2, p.z));
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
      // Lower tessellation — lookalike layout, not a mesh export.
      const geo = new THREE.TorusGeometry(Math.max(tube + 2, r - tube * 0.15), tube, 12, 48);
      const mesh = new THREE.Mesh(geo, sharedMat(engine, SI));
      const p = toWorld(d.x, d.y);
      mesh.position.set(p.x, siH / 2, p.z);
      mesh.rotation.x = Math.PI / 2;
      mesh.userData.deviceId = d.id;
      engine.pickables.push(mesh);
      engine.baseColors.set(mesh, SI);
      engine.deviceCenters.set(d.id, new THREE.Vector3(p.x, siH / 2, p.z));
      engine.root.add(mesh);
      const label = makeLabel(d.config === "add-drop" ? "ring (add-drop)" : "ring", "ring");
      label.position.set(p.x, siH + 24, p.z);
      engine.root.add(label);
    } else if (d.type === "heater") {
      const ringNear = nearestRing(d.x, d.y, devices);
      if (ringNear) {
        const r = ringPx(ringNear.radius_um);
        const arc = Math.min(4.7, Math.max(3.4, (2 * d.length_um) / Math.max(ringNear.radius_um, 6)));
        const geo = new THREE.TorusGeometry(r, Math.max(1.6, d.width_um * 0.85), 8, 36, arc);
        const mesh = new THREE.Mesh(geo, sharedMat(engine, HEATER));
        const p = toWorld(ringNear.x, ringNear.y);
        mesh.position.set(p.x, siH * 1.05, p.z);
        mesh.rotation.x = Math.PI / 2;
        mesh.rotation.z = Math.atan2(d.y - ringNear.y, d.x - ringNear.x) - arc / 2;
        mesh.userData.deviceId = d.id;
        engine.pickables.push(mesh);
        engine.baseColors.set(mesh, HEATER);
        engine.deviceCenters.set(d.id, new THREE.Vector3(p.x, siH * 1.05, p.z));
        engine.root.add(mesh);
        const lp = toWorld(d.x, d.y);
        const label = makeLabel("heater", "muted");
        label.position.set(lp.x, siH + 22, lp.z);
        engine.root.add(label);
      } else {
        const hw = Math.max(22, Math.min(40, d.length_um * 0.7));
        const hd = Math.max(5, Math.min(10, d.width_um * 3));
        addBox(d.x - hw / 2, d.y - hd / 2, hw, hd, siH * 0.85, siH * 0.45, HEATER, d.id);
        const p = toWorld(d.x, d.y);
        const label = makeLabel("heater", "muted");
        label.position.set(p.x, siH + 20, p.z);
        engine.root.add(label);
      }
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
  engine.span = span;
  if (doResetCamera) resetCamera(engine);
  else engine.invalidate(2);

  return { addDrop, cladName };
}

export function Chip3D({
  devices,
  selectedId,
  widthNm,
  heightNm,
  nClad,
  active = true,
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
  const activeRef = useRef(active);
  activeRef.current = active;

  const layoutKey = useMemo(
    () => layoutKeyOf(devices, widthNm, heightNm, nClad),
    [devices, widthNm, heightNm, nClad]
  );

  // One WebGL context for the life of the host (kept alive across view switches).
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
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
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

    const selectedMat = makeMat(ACCENT);
    selectedMat.emissive.setHex(ACCENT);
    selectedMat.emissiveIntensity = 0.22;
    selectedMat.userData.shared = true;

    let raf = 0;
    let framesLeft = 0;
    let interacting = false;

    const invalidate = (frames = 1) => {
      framesLeft = Math.max(framesLeft, frames);
      if (!raf) raf = requestAnimationFrame(tick);
    };

    const tick = () => {
      raf = 0;
      if (!activeRef.current || document.hidden) return;
      controls.update();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
      if (interacting || framesLeft > 0) {
        if (framesLeft > 0) framesLeft -= 1;
        raf = requestAnimationFrame(tick);
      }
    };

    const resize = () => {
      const el = hostRef.current;
      if (!el) return;
      const w = el.clientWidth || 480;
      const h = el.clientHeight || 360;
      if (w < 2 || h < 2) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
      labelRenderer.setSize(w, h);
      invalidate(2);
    };

    const ro = new ResizeObserver(() => resize());
    ro.observe(host);

    const onControlsChange = () => invalidate(2);
    const onControlsStart = () => {
      interacting = true;
      invalidate(2);
    };
    const onControlsEnd = () => {
      interacting = false;
      // Let damping settle without a permanent 60 fps loop.
      invalidate(48);
    };
    controls.addEventListener("change", onControlsChange);
    controls.addEventListener("start", onControlsStart);
    controls.addEventListener("end", onControlsEnd);

    const onVisibility = () => {
      if (!document.hidden && activeRef.current) invalidate(2);
    };
    document.addEventListener("visibilitychange", onVisibility);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let pointerDown = false;
    let dragged = false;
    let downX = 0;
    let downY = 0;

    function hitId(e: PointerEvent): string | null {
      const rect = renderer.domElement.getBoundingClientRect();
      if (!rect.width || !rect.height) return null;
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(engine.pickables, false);
      return (hits[0]?.object.userData.deviceId as string | undefined) ?? null;
    }

    function onPointerDown(e: PointerEvent) {
      pointerDown = true;
      dragged = false;
      downX = e.clientX;
      downY = e.clientY;
    }
    function onPointerMove(e: PointerEvent) {
      if (pointerDown) {
        if (Math.hypot(e.clientX - downX, e.clientY - downY) > 5) dragged = true;
        return;
      }
      renderer.domElement.style.cursor = hitId(e) ? "pointer" : "";
    }
    function onPointerUp(e: PointerEvent) {
      if (!pointerDown) return;
      pointerDown = false;
      if (dragged) return;
      onSelectRef.current(hitId(e));
      invalidate(2);
    }
    function onPointerLeave() {
      pointerDown = false;
      renderer.domElement.style.cursor = "";
    }

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("pointerleave", onPointerLeave);

    const engine: Engine = {
      scene,
      camera,
      renderer,
      labelRenderer,
      controls,
      root,
      pickables: [],
      baseColors: new Map(),
      sharedMats: new Map(),
      selectedMat,
      deviceCenters: new Map(),
      layoutKey: "",
      siH: 20,
      span: 200,
      invalidate,
      resize,
      dispose: () => {
        cancelAnimationFrame(raf);
        raf = 0;
        ro.disconnect();
        document.removeEventListener("visibilitychange", onVisibility);
        controls.removeEventListener("change", onControlsChange);
        controls.removeEventListener("start", onControlsStart);
        controls.removeEventListener("end", onControlsEnd);
        renderer.domElement.removeEventListener("pointerdown", onPointerDown);
        renderer.domElement.removeEventListener("pointermove", onPointerMove);
        renderer.domElement.removeEventListener("pointerup", onPointerUp);
        renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
        controls.dispose();
        clearGroup(root);
        for (const mat of engine.sharedMats.values()) mat.dispose();
        selectedMat.dispose();
        renderer.dispose();
        renderer.domElement.remove();
        labelRenderer.domElement.remove();
      },
    };
    engineRef.current = engine;
    resize();
    invalidate(2);

    return () => {
      engine.dispose();
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
    const doReset = engine.layoutKey === "";
    const meta = buildLayout(engine, devices, widthNm, heightNm, nClad, selectedRef.current, doReset);
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

  // Resume rendering + fix size when returning to the 3D tab.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    if (active) {
      engine.resize();
      engine.invalidate(4);
    }
  }, [active]);

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
        <div className="iso-webgl-toolbar">
          <button
            type="button"
            onClick={() => {
              const engine = engineRef.current;
              if (engine) resetCamera(engine);
            }}
          >
            Reset view
          </button>
          <button
            type="button"
            disabled={!selectedId}
            onClick={() => {
              const engine = engineRef.current;
              if (engine) frameSelection(engine, selectedId);
            }}
          >
            Frame selection
          </button>
        </div>
      </div>
      <p className="board-hint">
        Drag to orbit, scroll to zoom. Click a part to select it in Properties — layout look only, not
        a 3D FDTD mesh.
      </p>
    </div>
  );
}
