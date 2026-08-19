import { useEffect, useRef, useState, type PointerEvent, type ReactNode } from "react";

type RingConfig = "all-pass" | "add-drop";
type Polarization = "TE" | "TM";
type BoardView = "top" | "section";
type DeviceType = "ring" | "waveguide" | "coupler" | "detector";

type Platform = {
  wavelength_nm: number;
  width_nm: number;
  height_nm: number;
  n_clad: number;
  polarization: Polarization;
};

type RingDevice = {
  id: string;
  type: "ring";
  x: number;
  y: number;
  radius_um: number;
  config: RingConfig;
};

type WaveguideDevice = {
  id: string;
  type: "waveguide";
  x: number;
  y: number;
  length_um: number;
};

type CouplerDevice = {
  id: string;
  type: "coupler";
  x: number;
  y: number;
  gap_nm: number;
  length_um: number;
};

type DetectorDevice = {
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

type PlacedDevice = RingDevice | WaveguideDevice | CouplerDevice | DetectorDevice;

type FieldMap = {
  x_nm?: number[];
  y_nm?: number[];
  x_um?: number[];
  z_um?: number[];
  intensity: number[][];
  quantity: string;
  colormap?: "jet" | "density";
  layout?: "strip";
  core?: { x0: number; y0: number; width: number; height: number };
  guides?: { x0: number; y0: number; width: number; height: number }[];
  polygons?: { x: number; y: number }[][];
};

type SParams = {
  s11: number;
  s21: number;
  s31: number;
  s41: number;
  s11_db: number;
  s21_db: number;
  s31_db: number;
  s41_db: number;
  t_through: number;
  t_drop: number;
  t_through_db: number;
  t_drop_db: number | null;
};

type Extracted = {
  resonance_nm?: number;
  fwhm_nm?: number;
  q_loaded?: number;
  q_intrinsic?: number | null;
  q_coupling?: number;
  extinction_db?: number;
  insertion_loss_db?: number;
  t_through?: number;
  t_drop?: number | null;
  t_drop_db?: number | null;
  t_through_db?: number;
  kappa?: number;
  self_coupling_t?: number;
  fsr_nm?: number;
  beat_length_um?: number | null;
  imbalance_db?: number;
};

type CouplerResult = {
  kappa: number;
  through: number;
  cross: number;
  beat_length_um: number | null;
  t_through?: number;
  t_drop?: number;
  t_through_db?: number;
  t_drop_db?: number | null;
  s_parameters?: SParams;
  extracted?: Extracted;
  length_um?: number[];
  through_vs_length?: number[];
  cross_vs_length?: number[];
  wavelength_nm?: number[];
  through_vs_wavelength?: number[];
  cross_vs_wavelength?: number[];
  field?: FieldMap;
};

type ModeResult = {
  n_eff: number;
  n_g: number;
  dn_eff_dn: number;
  loss_db_per_cm: number;
  gamma_core: number;
  gamma_clad: number;
  t_through?: number;
  s_parameters?: SParams;
  field?: FieldMap;
};

type SpectrumData = {
  wavelength_nm: number[];
  through: number[];
  drop?: number[];
  t_through?: number[];
  t_drop?: number[];
  s21_db?: number[];
  s31_db?: number[];
};

type RingResult = {
  resonance_nm: number;
  mode_order: number;
  fsr_nm: number;
  q: number;
  q_intrinsic?: number | null;
  q_coupling?: number;
  fwhm_nm: number;
  finesse: number;
  coupling_regime: string;
  extinction_db: number;
  t_through?: number;
  t_drop?: number | null;
  t_through_db?: number;
  t_drop_db?: number | null;
  s_parameters?: SParams;
  extracted?: Extracted;
  sensitivity_nm_per_riu: number;
  lod_riu: number | null;
  spectrum: SpectrumData;
};

type CircuitResult = {
  mode: ModeResult;
  s_parameters: SParams;
  t_through: number;
  t_drop: number | null;
  t_through_db: number;
  t_drop_db: number | null;
  kappa: number | null;
  wg_loss_db: number;
  device_count: number;
  resonance_nm?: number;
  fsr_nm?: number;
  q?: number;
  q_intrinsic?: number | null;
  q_coupling?: number;
  fwhm_nm?: number;
  coupling_regime?: string;
  sensitivity_nm_per_riu?: number;
  lod_riu?: number | null;
  extinction_db?: number;
  extracted?: Extracted | null;
  spectrum?: SpectrumData | null;
  detectors?: (DetectorResult & { port: string; optical_power_uw: number })[];
  radius_for_laser_um?: number | null;
  analyte_sweep?: AnalyteSweep | null;
  critical?: CriticalCoupling | null;
  fdtd?: FdtdReport | null;
};

type FdtdReport = {
  geometry: { primitives: { kind: string; role?: string; material?: string }[]; gdsii: string; background: string };
  mesh: { kind: string; dx_nm: number; nx: number; ny: number; cells: number; conformal: boolean; subpixel: string; n_core_2d?: number };
  materials: { catalog: { name: string; n: number; model: string }[]; dispersion: string; complex_media: { anisotropic: boolean; nonlinear: boolean; gain: boolean } };
  boundaries: { absorbing: string; npml: number; symmetry_x: string | null; periodic: boolean; bloch: boolean; domain_reduction: string };
  sources: { mode_solver: string; injection: string; dipole: boolean; tfsf: boolean; polarization: string };
  engine: { scheme: string; courant: number; parallel: string; dimension: string };
  monitors: { dft: boolean; poynting: boolean; mode_overlap: boolean; far_field: { method: string; enabled: boolean }; ports: string[] };
  mode?: { n_eff?: number | null; method?: string; field?: FieldMap; error?: string } | null;
  run?: {
    steps?: number;
    t_through?: number;
    t_drop?: number | null;
    kappa?: number | null;
    overlap_through?: number;
    field?: FieldMap;
    engine?: string;
    error?: string;
  } | null;
};

type TopologyDrc = {
  mfs_nm: number;
  min_gap_nm: number;
  min_feature_nm: number;
  min_gap_meas_nm: number;
  mfs_ok: boolean;
  gap_ok: boolean;
  fill: number;
};

type TopologyResult = {
  error?: string;
  filter: string;
  projection: string;
  fabrication: string;
  vectorizer: string;
  dx_nm: number;
  beta: number;
  etas?: { dilated: number; intermediate: number; eroded: number };
  filter_radius_nm: number;
  steps: number;
  kappa_target: number | null;
  t_through: number | null;
  t_drop: number | null;
  kappa: number | null;
  grayscale: number;
  drc: TopologyDrc;
  history: {
    step: number;
    objective: number;
    t_drop: number;
    beta: number;
    worst?: "dilated" | "intermediate" | "eroded";
    t_drop_dilated?: number;
    t_drop_intermediate?: number;
    t_drop_eroded?: number;
  }[];
  robust?: {
    dilated?: { t_through: number; t_drop: number };
    intermediate?: { t_through: number; t_drop: number };
    eroded?: { t_through: number; t_drop: number };
    worst?: "dilated" | "intermediate" | "eroded";
    t_drop_worst?: number;
  } | null;
  polygons: { x: number; y: number }[][];
  polygon_count: number;
  gds_b64: string;
  gds_bytes: number;
  field: FieldMap;
  field_dilated?: FieldMap;
  field_eroded?: FieldMap;
  field_ez?: FieldMap;
};

type CriticalCoupling = {
  kappa: number;
  a: number;
  t: number;
  gap_nm: number | null;
  length_um: number;
  preferred: "gap" | "length";
  reachable: boolean;
};

type AnalyteSweep = {
  n_clad: number[];
  resonance_nm: number[];
  shift_nm: number[];
  t_through: number[];
  t_drop: number[] | null;
  sensitivity_nm_per_riu: number;
  n_design: number;
};

type DetectorResult = {
  photocurrent_ua: number;
  noise_current_na: number;
  snr_db: number;
  nep_pw_per_sqrt_hz: number | null;
  min_detectable_uw: number | null;
  quantum_efficiency: number;
};

const platformDefaults: Platform = {
  wavelength_nm: 1550,
  width_nm: 450,
  height_nm: 220,
  n_clad: 1.33,
  polarization: "TE",
};

const library: { type: DeviceType; label: string }[] = [
  { type: "waveguide", label: "Waveguide" },
  { type: "coupler", label: "Coupler" },
  { type: "ring", label: "Ring resonator" },
  { type: "detector", label: "Detector" },
];

function DeviceIcon({ type }: { type: DeviceType }) {
  return (
    <span className="lib-icon" aria-hidden="true">
      {type === "waveguide" ? (
        <svg viewBox="0 0 16 16">
          <line x1="1" y1="8" x2="15" y2="8" />
        </svg>
      ) : null}
      {type === "coupler" ? (
        <svg viewBox="0 0 16 16">
          <line x1="2" y1="6" x2="14" y2="6" />
          <line x1="2" y1="10" x2="14" y2="10" />
        </svg>
      ) : null}
      {type === "ring" ? (
        <svg viewBox="0 0 16 16">
          <circle cx="8" cy="8" r="4.5" />
        </svg>
      ) : null}
      {type === "detector" ? (
        <svg viewBox="0 0 16 16">
          <polygon points="3,3 13,8 3,13" />
          <line x1="13" y1="4" x2="13" y2="12" />
        </svg>
      ) : null}
    </span>
  );
}

let nextId = 1;
function uid(prefix: string) {
  nextId += 1;
  return `${prefix}-${nextId}`;
}

function waveguides(devices: PlacedDevice[]) {
  return devices.filter((d): d is WaveguideDevice => d.type === "waveguide");
}

function nearestBus(x: number, y: number, buses: WaveguideDevice[]) {
  if (!buses.length) return null;
  return buses.reduce((best, b) => {
    const dBest = (best.x - x) ** 2 + (best.y - y) ** 2;
    const dB = (b.x - x) ** 2 + (b.y - y) ** 2;
    return dB < dBest ? b : best;
  });
}

const CHIP_W = 520;
const CHIP_H = 340;
const BUS_X = 260;
const THROUGH_Y = 210;

function wgHalf(lengthUm: number) {
  return Math.max(50, Math.min(150, lengthUm * 0.7)) / 2;
}

function ringPx(radiusUm: number) {
  return Math.max(18, Math.min(34, radiusUm * 2.2));
}

function clampChip(x: number, y: number) {
  return {
    x: Math.min(CHIP_W - 24, Math.max(24, x)),
    y: Math.min(CHIP_H - 24, Math.max(24, y)),
  };
}

function arrange(devices: PlacedDevice[]): PlacedDevice[] {
  if (!devices.length) return devices;
  const rings = devices.filter((d): d is RingDevice => d.type === "ring");
  const buses = waveguides(devices);
  const couplers = devices.filter((d): d is CouplerDevice => d.type === "coupler");
  const dets = devices.filter((d): d is DetectorDevice => d.type === "detector");
  const addDrop = rings.some((r) => r.config === "add-drop");
  const pos = new Map<string, { x: number; y: number }>();

  buses.forEach((b, i) => {
    pos.set(b.id, clampChip(BUS_X, THROUGH_Y + (i === 0 ? 0 : 70 * i)));
  });

  rings.forEach((r, i) => {
    const ty = buses[0] ? pos.get(buses[0].id)!.y : THROUGH_Y;
    pos.set(r.id, clampChip(250 + i * 96, ty - ringPx(r.radius_um) - 10));
  });

  const ad = rings.find((r) => r.config === "add-drop");
  if (ad && buses[1]) {
    const rp = pos.get(ad.id)!;
    pos.set(buses[1].id, clampChip(BUS_X, rp.y - ringPx(ad.radius_um) - 10));
  }

  const throughCouplers = addDrop ? [couplers[0], ...couplers.slice(2)] : couplers;
  throughCouplers.forEach((c, i) => {
    if (!c) return;
    const ring = rings[i] ?? rings[0];
    const x = ring ? pos.get(ring.id)!.x : 250;
    const y = buses[0] ? pos.get(buses[0].id)!.y : THROUGH_Y;
    pos.set(c.id, clampChip(x, y));
  });
  if (addDrop && couplers[1] && buses[1]) {
    const ring = rings[0];
    pos.set(
      couplers[1].id,
      clampChip(ring ? pos.get(ring.id)!.x : 250, pos.get(buses[1].id)!.y)
    );
  }

  dets.forEach((d, i) => {
    const bus = buses[Math.min(i, Math.max(0, buses.length - 1))];
    if (!bus) {
      pos.set(d.id, clampChip(430, THROUGH_Y));
      return;
    }
    const bp = pos.get(bus.id)!;
    pos.set(d.id, clampChip(bp.x + wgHalf(bus.length_um) + 18, bp.y));
  });

  return devices.map((d) => {
    const p = pos.get(d.id);
    return p ? ({ ...d, ...p } as PlacedDevice) : d;
  });
}

function withPrereqs(existing: PlacedDevice[], type: DeviceType) {
  const next = [...existing];
  if (
    (type === "ring" || type === "coupler" || type === "detector") &&
    !waveguides(next).length
  ) {
    next.push(createDevice("waveguide", next));
  }
  if (type === "ring" && !next.some((d) => d.type === "coupler")) {
    next.push(createDevice("coupler", next));
  }
  return next;
}

function waveguideRole(wg: WaveguideDevice, devices: PlacedDevice[]) {
  const buses = waveguides(devices);
  const ring = devices.find((d): d is RingDevice => d.type === "ring" && d.config === "add-drop");
  if (ring && buses.length >= 2) {
    if (wg.y < ring.y) return { caption: "drop", left: "add", right: "drop" };
    return { caption: "through", left: "in", right: "through" };
  }
  if (buses.length > 1) {
    const n = buses.findIndex((b) => b.id === wg.id) + 1;
    return { caption: `waveguide ${n}`, left: "in", right: "out" };
  }
  return { caption: "waveguide", left: "in", right: "out" };
}

function place(type: DeviceType, existing: PlacedDevice[]) {
  const buses = waveguides(existing);
  const ring = existing.find((d): d is RingDevice => d.type === "ring");
  const bus = ring ? nearestBus(ring.x, ring.y, buses) : buses[0];
  if (type === "waveguide") {
    if (!buses.length) return { x: BUS_X, y: THROUGH_Y };
    return { x: BUS_X, y: THROUGH_Y + 70 * buses.length };
  }
  if (type === "coupler") return { x: bus?.x ?? 250, y: bus?.y ?? THROUGH_Y };
  if (type === "ring") {
    return { x: bus?.x ?? 250, y: (bus?.y ?? THROUGH_Y) - 76 };
  }
  return { x: (bus?.x ?? BUS_X) + 170, y: bus?.y ?? THROUGH_Y };
}

function starterDevices(): PlacedDevice[] {
  const next: PlacedDevice[] = [];
  next.push(createDevice("waveguide", next));
  next.push(createDevice("coupler", next));
  next.push(createDevice("ring", next));
  next.push(createDevice("detector", next));
  return arrange(next);
}

function nearestRing(x: number, y: number, devices: PlacedDevice[]) {
  const rings = devices.filter((d): d is RingDevice => d.type === "ring");
  if (!rings.length) return null;
  return rings.reduce((best, r) => {
    const dBest = (best.x - x) ** 2 + (best.y - y) ** 2;
    const dR = (r.x - x) ** 2 + (r.y - y) ** 2;
    return dR < dBest ? r : best;
  });
}

function snapPos(
  x: number,
  y: number,
  type: DeviceType,
  others: PlacedDevice[],
  gridOn: boolean
) {
  if (gridOn) {
    x = Math.round(x / 8) * 8;
    y = Math.round(y / 8) * 8;
  }
  const buses = waveguides(others);
  if (type === "waveguide") {
    let aligned = x;
    let bestDx = 16;
    for (const bus of buses) {
      const dx = Math.abs(x - bus.x);
      if (dx < bestDx) {
        bestDx = dx;
        aligned = bus.x;
      }
    }
    let yy = y;
    for (const bus of buses) {
      if (Math.abs(yy - bus.y) < 40) yy = yy >= bus.y ? bus.y + 48 : bus.y - 48;
    }
    return { x: aligned, y: yy };
  }
  if (type === "coupler" || type === "detector") {
    let best: WaveguideDevice | null = null;
    let bestDy = 20;
    for (const bus of buses) {
      const dy = Math.abs(y - bus.y);
      if (dy < bestDy) {
        bestDy = dy;
        best = bus;
      }
    }
    return { x, y: best ? best.y : y };
  }
  let dock: number | null = null;
  let best = 20;
  for (const bus of buses) {
    for (const yDock of [bus.y - 76, bus.y + 76]) {
      const dy = Math.abs(y - yDock);
      if (dy < best) {
        best = dy;
        dock = yDock;
      }
    }
  }
  return { x, y: dock ?? y };
}

function createDevice(type: DeviceType, existing: PlacedDevice[]): PlacedDevice {
  const p = place(type, existing);
  if (type === "waveguide") return { id: uid("wg"), type, ...p, length_um: 180 };
  if (type === "coupler") return { id: uid("cpl"), type, ...p, gap_nm: 200, length_um: 12 };
  if (type === "ring") return { id: uid("ring"), type, ...p, radius_um: 10, config: "all-pass" };
  return {
    id: uid("pd"),
    type: "detector",
    ...p,
    optical_power_uw: 10,
    responsivity_a_per_w: 0.9,
    dark_current_na: 5,
    bandwidth_mhz: 100,
    load_ohm: 50,
  };
}

function labelOf(d: PlacedDevice) {
  if (d.type === "waveguide") return "Waveguide";
  if (d.type === "coupler") return "Coupler";
  if (d.type === "ring") return "Ring";
  return "Detector";
}

function fmt(value: number | null | undefined, digits = 3) {
  if (value == null || Number.isNaN(value)) return "—";
  if (Math.abs(value) >= 1e6 || (value !== 0 && Math.abs(value) < 1e-3)) {
    return value.toExponential(2);
  }
  return Number(value).toFixed(digits);
}

export default function App() {
  const [platform, setPlatform] = useState<Platform>(platformDefaults);
  const [devices, setDevices] = useState<PlacedDevice[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mode, setMode] = useState<ModeResult | null>(null);
  const [circuit, setCircuit] = useState<CircuitResult | null>(null);
  const [error, setError] = useState("");
  const [solveStatus, setSolveStatus] = useState<"solving" | "ready" | "error">("solving");
  const [grid, setGrid] = useState(true);
  const [view, setView] = useState<BoardView>("top");
  const [libraryOpen, setLibraryOpen] = useState(true);
  const [propertiesOpen, setPropertiesOpen] = useState(true);
  const [resultsOpen, setResultsOpen] = useState(true);
  const [fdtdRun, setFdtdRun] = useState<FdtdReport | null>(null);
  const [fdtdBusy, setFdtdBusy] = useState(false);
  const [topologyRun, setTopologyRun] = useState<TopologyResult | null>(null);
  const [topologyBusy, setTopologyBusy] = useState(false);

  const selected = devices.find((d) => d.id === selectedId) ?? null;

  function setPlatformField<K extends keyof Platform>(name: K, value: Platform[K]) {
    setPlatform((prev) => ({ ...prev, [name]: value }));
  }

  function addDevice(type: DeviceType) {
    const built = withPrereqs(devices, type);
    const device = createDevice(type, built);
    setDevices(arrange([...built, device]));
    setSelectedId(device.id);
    setView("top");
  }

  function placeSensor() {
    const next = starterDevices();
    setDevices(next);
    setSelectedId(next.find((d) => d.type === "ring")?.id ?? next[0].id);
    setView("top");
  }

  function clearCircuit() {
    setDevices([]);
    setSelectedId(null);
  }

  function removeSelected() {
    if (!selectedId) return;
    setDevices((prev) => arrange(prev.filter((d) => d.id !== selectedId)));
    setSelectedId(null);
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Backspace" && e.key !== "Delete") return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      e.preventDefault();
      removeSelected();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId]);

  function patchDevice(id: string, patch: Partial<PlacedDevice>) {
    setDevices((prev) => {
      const next = prev.map((d) => (d.id === id ? ({ ...d, ...patch } as PlacedDevice) : d));
      if (patch.x != null || patch.y != null) return next;
      return arrange(next);
    });
  }

  function patchSelected(patch: Partial<PlacedDevice>) {
    if (!selectedId) return;
    patchDevice(selectedId, patch);
  }

  function tuneRingToLaser() {
    const r = circuit?.radius_for_laser_um;
    if (r == null) return;
    const ring =
      selected?.type === "ring" ? selected : devices.find((d) => d.type === "ring");
    if (!ring) return;
    patchDevice(ring.id, { radius_um: Math.round(r * 1e4) / 1e4 });
  }

  function setCriticalCoupling() {
    const c = circuit?.critical;
    if (!c) return;
    const patch =
      c.preferred === "gap" && c.gap_nm != null
        ? { gap_nm: Math.round(c.gap_nm * 10) / 10 }
        : { length_um: Math.round(c.length_um * 100) / 100 };
    setDevices((prev) =>
      arrange(prev.map((d) => (d.type === "coupler" ? { ...d, ...patch } : d)))
    );
  }

  async function runTopology() {
    setTopologyBusy(true);
    try {
      const pin = devices.find((d): d is DetectorDevice => d.type === "detector");
      const res = await fetch("/api/topology", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...platform,
          devices,
          input_power_uw: pin?.optical_power_uw ?? 10,
          kappa_target: circuit?.critical?.kappa ?? circuit?.kappa ?? undefined,
        }),
      });
      if (!res.ok) {
        setError("Topology run failed.");
        return;
      }
      setTopologyRun((await res.json()) as TopologyResult);
    } catch {
      setError("Could not reach the Python backend. Start it on port 8000.");
    } finally {
      setTopologyBusy(false);
    }
  }

  async function runFdtd() {
    setFdtdBusy(true);
    try {
      const pin = devices.find((d): d is DetectorDevice => d.type === "detector");
      const res = await fetch("/api/fdtd", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...platform,
          devices,
          input_power_uw: pin?.optical_power_uw ?? 10,
        }),
      });
      if (!res.ok) {
        setError("2D FDTD run failed.");
        return;
      }
      setFdtdRun((await res.json()) as FdtdReport);
    } catch {
      setError("Could not reach the Python backend. Start it on port 8000.");
    } finally {
      setFdtdBusy(false);
    }
  }

  function applyRingConfig(config: RingConfig) {
    if (!selected || selected.type !== "ring") return;
    const ringId = selected.id;
    setDevices((prev) => {
      let next = prev.map((d) =>
        d.id === ringId && d.type === "ring" ? { ...d, config } : d
      );
      if (config === "add-drop") {
        if (waveguides(next).length < 2) next = [...next, createDevice("waveguide", next)];
        if (next.filter((d) => d.type === "coupler").length < 2) {
          next = [...next, createDevice("coupler", next)];
        }
      }
      return arrange(next);
    });
  }

  // Whole design — not the clicked part
  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setSolveStatus("solving");
      try {
        const pin = devices.find((d): d is DetectorDevice => d.type === "detector");
        const res = await fetch("/api/circuit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...platform,
            devices,
            input_power_uw: pin?.optical_power_uw ?? 10,
          }),
          signal: controller.signal,
        });
        if (!res.ok) {
          setMode(null);
          setCircuit(null);
          setError("No guided mode for this cross-section.");
          setSolveStatus("error");
          return;
        }
        const data = (await res.json()) as CircuitResult;
        setCircuit(data);
        setMode(data.mode);
        setError("");
        setSolveStatus("ready");
        setFdtdRun(null);
        setTopologyRun(null);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError("Could not reach the Python backend. Start it on port 8000.");
        setSolveStatus("error");
      }
    }, 120);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [devices, platform]);

  const leftOpen = libraryOpen || propertiesOpen;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg className="logo" viewBox="0 0 32 32" aria-hidden="true">
            <rect className="logo-bg" x="0" y="0" width="32" height="32" rx="8" />
            <g className="logo-mark">
              <line x1="4" y1="16" x2="28" y2="16" />
              <circle cx="16" cy="16" r="6.5" />
            </g>
          </svg>
          <div className="brand-text">
            <h1>Photonic sensor</h1>
            <p>SOI compact models</p>
          </div>
        </div>
        <div className="globals">
          <p className={`solve-status ${solveStatus}`} role="status">
            {solveStatus === "solving"
              ? "Solving…"
              : solveStatus === "ready"
                ? "Up to date"
                : error.includes("Could not reach")
                  ? "Backend unreachable"
                  : error.includes("No guided mode")
                    ? "No guided mode"
                    : error || "Error"}
          </p>
          <label className="global-field">
            λ (nm)
            <input
              type="number"
              value={platform.wavelength_nm}
              onChange={(e) => setPlatformField("wavelength_nm", Number(e.target.value))}
            />
          </label>
          {circuit?.radius_for_laser_um != null || circuit?.critical ? (
            <div className="toolbar-actions">
              {circuit?.radius_for_laser_um != null ? (
                <button type="button" onClick={tuneRingToLaser}>
                  Tune ring to λ
                </button>
              ) : null}
              {circuit?.critical ? (
                <button type="button" onClick={setCriticalCoupling}>
                  Set critical κ
                </button>
              ) : null}
            </div>
          ) : null}
          <div className="view-toggle">
            <button
              type="button"
              className={view === "top" ? "active" : ""}
              onClick={() => setView("top")}
            >
              Circuit
            </button>
            <button
              type="button"
              className={view === "section" ? "active" : ""}
              onClick={() => setView("section")}
            >
              Cross-section
            </button>
          </div>
          <label className="switch">
            <input
              type="checkbox"
              checked={grid}
              onChange={(e) => setGrid(e.target.checked)}
            />
            <span className="switch-ui" />
            Grid
          </label>
        </div>
      </header>

      <div className="workspace">
        <div className={`left-stack ${leftOpen ? "open" : "closed"}`}>
          <Bar
            side="left"
            title="Library"
            open={libraryOpen}
            stacked
            onToggle={() => setLibraryOpen((v) => !v)}
          >
            <p className="bar-section">Start</p>
            <div className="library-list">
              {devices.length === 0 ? (
                <button type="button" className="primary" onClick={placeSensor}>
                  Place a ring sensor
                </button>
              ) : (
                <button type="button" onClick={clearCircuit}>
                  Clear chip
                </button>
              )}
            </div>
            <p className="bar-section">Add a part</p>
            <div className="library-list">
              {library.map((item) => (
                <button
                  key={item.type}
                  type="button"
                  className="lib-item"
                  onClick={() => addDevice(item.type)}
                >
                  <DeviceIcon type={item.type} />
                  {item.label}
                </button>
              ))}
            </div>
            <p className="muted">
              Parts are placed automatically. Drag only if you want to nudge them.
            </p>
          </Bar>

          <Bar
            side="left"
            title="Properties"
            open={propertiesOpen}
            stacked
            grow
            onToggle={() => setPropertiesOpen((v) => !v)}
          >
            <div className="prop-block">
              <p className="bar-section">Platform</p>
              <Field
                label="Width (nm)"
                value={platform.width_nm}
                onChange={(v) => setPlatformField("width_nm", v)}
                step="10"
                min={250}
                max={800}
              />
              <Field
                label="Height (nm)"
                value={platform.height_nm}
                onChange={(v) => setPlatformField("height_nm", v)}
                step="10"
                min={150}
                max={340}
              />
              <label>
                Cladding
                <select
                  value={platform.n_clad}
                  onChange={(e) => setPlatformField("n_clad", Number(e.target.value))}
                >
                  <option value={1}>Air (1.00)</option>
                  <option value={1.33}>Water (1.33)</option>
                  <option value={1.444}>Oxide (1.444)</option>
                </select>
              </label>
              <label>
                Polarization
                <select
                  value={platform.polarization}
                  onChange={(e) =>
                    setPlatformField("polarization", e.target.value as Polarization)
                  }
                >
                  <option value="TE">TE</option>
                  <option value="TM">TM</option>
                </select>
              </label>
              {mode ? (
                <>
                  <div className="metrics">
                    <Metric label="n_eff" value={fmt(mode.n_eff)} />
                    <Metric label="n_g" value={fmt(mode.n_g)} />
                  </div>
                  <p className="muted">FDTD-fitted 220 nm SOI compact model.</p>
                </>
              ) : null}
              {error ? <p className="error">{error}</p> : null}
            </div>

            {selected ? (
              <div className="prop-block">
                <p className="bar-section">{labelOf(selected)}</p>
                {selected.type === "waveguide" ? (
                  <>
                    <p className="muted">
                      {waveguideRole(selected, devices).caption === "drop"
                        ? "Drop bus — opposite the through waveguide."
                        : waveguideRole(selected, devices).caption === "through"
                          ? "Through bus — the input waveguide."
                          : waveguides(devices).length > 1
                            ? "Each waveguide is its own bus. Couplers and detectors snap to the nearest one."
                            : "The bus waveguide. Add a second one for an add-drop ring."}
                    </p>
                    <Field
                      label="Length (μm)"
                      value={selected.length_um}
                      onChange={(v) => patchSelected({ length_um: v })}
                      step="1"
                      min={1}
                      max={5000}
                    />
                  </>
                ) : null}
                {selected.type === "coupler" ? (
                  <>
                    <Field
                      label="Gap (nm)"
                      value={selected.gap_nm}
                      onChange={(v) => patchSelected({ gap_nm: v })}
                      step="10"
                      min={50}
                      max={500}
                    />
                    <Field
                      label="Coupling length (μm)"
                      value={selected.length_um}
                      onChange={(v) => patchSelected({ length_um: v })}
                      step="0.5"
                      min={1}
                      max={200}
                    />
                    {circuit?.critical ? (
                      <>
                        <p className="muted">
                          Critical κ is {fmt(circuit.critical.kappa, 4)} (a ={" "}
                          {fmt(circuit.critical.a, 4)}).
                          {circuit.critical.preferred === "gap" && circuit.critical.gap_nm != null
                            ? ` Gap ${fmt(circuit.critical.gap_nm, 0)} nm at this length.`
                            : ` Length ${fmt(circuit.critical.length_um, 2)} μm at this gap.`}
                        </p>
                        <button type="button" onClick={setCriticalCoupling}>
                          Set critical κ
                        </button>
                      </>
                    ) : null}
                  </>
                ) : null}
                {selected.type === "ring" ? (
                  <>
                    <p className="muted">
                      Add-drop uses two waveguides: through below the ring, drop above it.
                    </p>
                    <label>
                      Configuration
                      <select
                        value={selected.config}
                        onChange={(e) => applyRingConfig(e.target.value as RingConfig)}
                      >
                        <option value="all-pass">All-pass</option>
                        <option value="add-drop">Add-drop</option>
                      </select>
                    </label>
                    <Field
                      label="Radius (μm)"
                      value={selected.radius_um}
                      onChange={(v) => patchSelected({ radius_um: v })}
                      step="0.1"
                      min={1}
                      max={80}
                    />
                    {circuit?.radius_for_laser_um != null ? (
                      <p className="muted">
                        Radius for {platform.wavelength_nm} nm is{" "}
                        {fmt(circuit.radius_for_laser_um, 4)} μm.
                        {circuit.resonance_nm != null
                          ? ` Current λ0 is ${fmt(circuit.resonance_nm, 4)} nm.`
                          : ""}
                      </p>
                    ) : null}
                    <button type="button" onClick={tuneRingToLaser}>
                      Tune ring to λ
                    </button>
                    {circuit?.critical ? (
                      <button type="button" onClick={setCriticalCoupling}>
                        Set critical κ
                      </button>
                    ) : null}
                  </>
                ) : null}
                {selected.type === "detector" ? (
                  <>
                    <Field
                      label="Optical power (μW)"
                      value={selected.optical_power_uw}
                      onChange={(v) => patchSelected({ optical_power_uw: v })}
                      step="0.1"
                      min={0.001}
                    />
                    <Field
                      label="Responsivity (A/W)"
                      value={selected.responsivity_a_per_w}
                      onChange={(v) => patchSelected({ responsivity_a_per_w: v })}
                      step="0.05"
                      min={0.05}
                      max={1.2}
                    />
                    <Field
                      label="Dark current (nA)"
                      value={selected.dark_current_na}
                      onChange={(v) => patchSelected({ dark_current_na: v })}
                      step="0.5"
                      min={0}
                    />
                    <Field
                      label="Bandwidth (MHz)"
                      value={selected.bandwidth_mhz}
                      onChange={(v) => patchSelected({ bandwidth_mhz: v })}
                      step="1"
                      min={0.1}
                    />
                    <Field
                      label="Load (Ω)"
                      value={selected.load_ohm}
                      onChange={(v) => patchSelected({ load_ohm: v })}
                      step="1"
                      min={1}
                    />
                  </>
                ) : null}
                <button type="button" className="danger" onClick={removeSelected}>
                  Remove from circuit
                </button>
              </div>
            ) : (
              <p className="muted">Click a part on the chip to edit it here.</p>
            )}
          </Bar>
        </div>

        <main className={`board ${grid ? "grid" : ""}`}>
          {view === "section" ? (
            <CrossSection
              widthNm={platform.width_nm}
              heightNm={platform.height_nm}
              nClad={platform.n_clad}
              field={mode?.field}
              onWidthChange={(v) => setPlatformField("width_nm", v)}
              onHeightChange={(v) => setPlatformField("height_nm", v)}
            />
          ) : (
            <CircuitBoard
              devices={devices}
              selectedId={selectedId}
              grid={grid}
              onSelect={setSelectedId}
              onPatch={patchDevice}
              onPlaceSensor={placeSensor}
            />
          )}
        </main>

        <Bar
          side="right"
          title="Results"
          open={resultsOpen}
          onToggle={() => setResultsOpen((v) => !v)}
        >
          {!circuit && solveStatus === "solving" ? (
            <p className="muted">Solving the circuit…</p>
          ) : null}
          {!circuit && solveStatus === "error" ? <p className="error">{error}</p> : null}
          {circuit && circuit.device_count === 0 ? (
            <p className="muted">Place a ring sensor to get S-parameters and T_drop.</p>
          ) : null}
          {circuit && circuit.device_count > 0 ? (
            <>
              <p className="bar-section">At laser λ</p>
              <div className="metrics">
                <Metric
                  label="λ0"
                  value={
                    circuit.resonance_nm != null ? `${fmt(circuit.resonance_nm, 4)} nm` : "—"
                  }
                />
                <Metric label="Q" value={circuit.q != null ? fmt(circuit.q, 0) : "—"} />
                <Metric label="T_through" value={fmt(circuit.t_through, 4)} />
                <Metric label="T_drop" value={fmt(circuit.t_drop, 4)} />
                <Metric
                  label="κ"
                  value={circuit.kappa != null ? fmt(circuit.kappa, 3) : "—"}
                />
              </div>
              <ResultFold title="Design" defaultOpen>
                <div className="metrics">
                  <Metric label="Devices" value={circuit.device_count} />
                  <Metric label="WG loss" value={`${fmt(circuit.wg_loss_db, 3)} dB`} />
                  {circuit.critical ? (
                    <Metric label="κ_crit" value={fmt(circuit.critical.kappa, 4)} />
                  ) : null}
                  <Metric
                    label="Sensitivity"
                    value={
                      circuit.sensitivity_nm_per_riu != null
                        ? `${fmt(circuit.sensitivity_nm_per_riu, 1)} nm/RIU`
                        : "—"
                    }
                  />
                  {circuit.resonance_nm != null ? (
                    <Metric
                      label="λ0 − λ"
                      value={`${fmt(circuit.resonance_nm - platform.wavelength_nm, 4)} nm`}
                    />
                  ) : null}
                  {circuit.fsr_nm != null ? (
                    <Metric label="FSR" value={`${fmt(circuit.fsr_nm)} nm`} />
                  ) : null}
                  {circuit.coupling_regime ? (
                    <Metric label="Coupling" value={circuit.coupling_regime} />
                  ) : null}
                </div>
              </ResultFold>
              <SParamTable s={circuit.s_parameters} />
              {circuit.extracted ? (
                <ResultFold title="Extracted parameters">
                  <div className="metrics">
                    <Metric
                      label="λ0"
                      value={`${fmt(circuit.extracted.resonance_nm, 4)} nm`}
                    />
                    <Metric
                      label="FWHM"
                      value={`${fmt(circuit.extracted.fwhm_nm, 4)} nm`}
                    />
                    <Metric label="Q_loaded" value={fmt(circuit.extracted.q_loaded, 0)} />
                    <Metric label="Q_i" value={fmt(circuit.extracted.q_intrinsic, 0)} />
                    <Metric label="Q_c" value={fmt(circuit.extracted.q_coupling, 0)} />
                    <Metric
                      label="ER"
                      value={`${fmt(circuit.extracted.extinction_db, 1)} dB`}
                    />
                  </div>
                </ResultFold>
              ) : null}
              {circuit.spectrum?.wavelength_nm?.length ? (
                <ResultFold title="Spectrum">
                  <LinePlot
                    title="S-parameters"
                    xLabel="Wavelength (nm)"
                    yLabel="|S| (dB)"
                    x={circuit.spectrum.wavelength_nm}
                    series={[
                      {
                        name: "|S21| through",
                        y: circuit.spectrum.through,
                        kind: "through",
                      },
                      ...(circuit.spectrum.drop
                        ? [
                            {
                              name: "|S31| drop",
                              y: circuit.spectrum.drop,
                              kind: "drop" as const,
                            },
                          ]
                        : []),
                    ]}
                    db
                    markX={platform.wavelength_nm}
                  />
                </ResultFold>
              ) : null}
              {circuit.analyte_sweep?.n_clad?.length ? (
                <ResultFold title="Analyte sweep">
                  <div className="metrics">
                    <Metric
                      label="dλ/dn"
                      value={`${fmt(circuit.analyte_sweep.sensitivity_nm_per_riu, 1)} nm/RIU`}
                    />
                  </div>
                  <LinePlot
                    title="Resonance vs cladding"
                    xLabel="n_clad"
                    yLabel="Δλ (nm)"
                    x={circuit.analyte_sweep.n_clad}
                    series={[
                      {
                        name: "λ0 − λ0(design)",
                        y: circuit.analyte_sweep.shift_nm,
                        kind: "through",
                      },
                    ]}
                    markX={circuit.analyte_sweep.n_design}
                    xDigits={3}
                    yAuto
                  />
                  <LinePlot
                    title="T at laser λ"
                    xLabel="n_clad"
                    yLabel="T"
                    x={circuit.analyte_sweep.n_clad}
                    series={[
                      {
                        name: "T_through (λ)",
                        y: circuit.analyte_sweep.t_through,
                        kind: "through",
                      },
                      ...(circuit.analyte_sweep.t_drop
                        ? [
                            {
                              name: "T_drop (λ)",
                              y: circuit.analyte_sweep.t_drop,
                              kind: "drop" as const,
                            },
                          ]
                        : []),
                    ]}
                    markX={circuit.analyte_sweep.n_design}
                    xDigits={3}
                  />
                </ResultFold>
              ) : null}
              {circuit.detectors?.length ? (
                <ResultFold title="Detectors" defaultOpen>
                  {circuit.detectors.map((d, i) => (
                    <div className="metrics" key={`${d.port}-${i}`}>
                      <Metric label="Port" value={d.port} />
                      <Metric label="P_opt" value={`${fmt(d.optical_power_uw, 3)} μW`} />
                      <Metric label="Photocurrent" value={`${fmt(d.photocurrent_ua, 3)} μA`} />
                      <Metric label="SNR" value={`${fmt(d.snr_db, 1)} dB`} />
                    </div>
                  ))}
                </ResultFold>
              ) : null}
              {circuit.mode.field ? (
                <ResultFold title="Mode field">
                  <Heatmap
                    title="Mode field monitor  |E|²"
                    xLabel="x (nm)"
                    yLabel="y (nm)"
                    field={circuit.mode.field}
                  />
                </ResultFold>
              ) : null}
              {circuit.fdtd ? (
                <ResultFold title="FDTD">
                  <FdtdPanel
                    report={fdtdRun ?? circuit.fdtd}
                    compactNeff={circuit.mode.n_eff}
                    busy={fdtdBusy}
                    onRun={runFdtd}
                  />
                </ResultFold>
              ) : null}
              <ResultFold title="Topology">
                <TopologyPanel
                  report={topologyRun}
                  busy={topologyBusy}
                  onRun={runTopology}
                />
              </ResultFold>
            </>
          ) : null}
        </Bar>
      </div>
    </div>
  );
}

function CircuitBoard({
  devices,
  selectedId,
  grid,
  onSelect,
  onPatch,
  onPlaceSensor,
}: {
  devices: PlacedDevice[];
  selectedId: string | null;
  grid: boolean;
  onSelect: (id: string | null) => void;
  onPatch: (id: string, patch: Partial<PlacedDevice>) => void;
  onPlaceSensor: () => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ id: string; type: DeviceType; dx: number; dy: number } | null>(null);
  const [dragging, setDragging] = useState(false);

  function localPoint(e: PointerEvent<SVGElement>) {
    if (!svgRef.current) return null;
    return toSvgPoint(e, svgRef.current);
  }

  function move(e: PointerEvent<SVGElement>) {
    if (!drag.current) return;
    const p = localPoint(e);
    if (!p) return;
    const rawX = Math.min(CHIP_W - 24, Math.max(24, p.x - drag.current.dx));
    const rawY = Math.min(CHIP_H - 24, Math.max(24, p.y - drag.current.dy));
    const others = devices.filter((d) => d.id !== drag.current!.id);
    const snapped = snapPos(rawX, rawY, drag.current.type, others, grid);
    onPatch(drag.current.id, snapped);
  }

  function endDrag() {
    drag.current = null;
    setDragging(false);
  }

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
        <p className="muted">Add a part and it will sit in a typical sensor layout.</p>
        <button type="button" className="primary" onClick={onPlaceSensor}>
          Place a ring sensor
        </button>
      </div>
    );
  }

  return (
    <div className="board-stage">
      <svg
        ref={svgRef}
        className={`circuit-svg ${dragging ? "dragging" : ""}`}
        viewBox={`0 0 ${CHIP_W} ${CHIP_H}`}
        preserveAspectRatio="xMidYMid meet"
        onPointerDown={() => onSelect(null)}
        onPointerMove={move}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        <rect className="chip" x={16} y={16} width={CHIP_W - 32} height={CHIP_H - 32} rx={10} />
        <text x={28} y={36}>
          chip
        </text>

        {devices.map((d) => {
          const selected = selectedId === d.id;
          const role = d.type === "waveguide" ? waveguideRole(d, devices) : null;
          const wgHalfLen = d.type === "waveguide" ? wgHalf(d.length_um) : 0;
          const cplHalf = d.type === "coupler" ? Math.max(28, Math.min(50, d.length_um * 3)) : 36;
          const cplGap = d.type === "coupler" ? Math.max(8, Math.min(16, d.gap_nm * 0.04)) : 12;
          const r = d.type === "ring" ? ringPx(d.radius_um) : 22;
          const ringNear = d.type === "coupler" ? nearestRing(d.x, d.y, devices) : null;
          const gapDir = ringNear && ringNear.y > d.y ? cplGap : -cplGap;

          return (
            <g
              key={d.id}
              className={`seg ${selected ? "selected" : ""}`}
              onPointerDown={(e) => {
                e.stopPropagation();
                onSelect(d.id);
                const p = localPoint(e);
                if (!p) return;
                drag.current = { id: d.id, type: d.type, dx: p.x - d.x, dy: p.y - d.y };
                setDragging(true);
                e.currentTarget.setPointerCapture(e.pointerId);
              }}
            >
              {d.type === "waveguide" ? (
                <>
                  {selected ? (
                    <rect
                      className="halo"
                      x={d.x - wgHalfLen - 8}
                      y={d.y - 14}
                      width={wgHalfLen * 2 + 16}
                      height={28}
                      rx={4}
                    />
                  ) : null}
                  <line className="hit" x1={d.x - wgHalfLen} y1={d.y} x2={d.x + wgHalfLen} y2={d.y} />
                  <line className="bus" x1={d.x - wgHalfLen} y1={d.y} x2={d.x + wgHalfLen} y2={d.y} />
                  <text className="io" x={d.x - wgHalfLen} y={d.y - 10} textAnchor="start">
                    {role?.left}
                  </text>
                  <text className="io" x={d.x + wgHalfLen} y={d.y - 10} textAnchor="end">
                    {role?.right}
                  </text>
                  <text className="caption" x={d.x} y={d.y + 20} textAnchor="middle">
                    {role?.caption}
                  </text>
                </>
              ) : null}

              {d.type === "coupler" ? (
                <>
                  {selected ? (
                    <rect
                      className="halo"
                      x={d.x - cplHalf - 8}
                      y={d.y + Math.min(0, gapDir) - 14}
                      width={cplHalf * 2 + 16}
                      height={Math.abs(gapDir) + 28}
                      rx={4}
                    />
                  ) : null}
                  <rect
                    className="hit-fill"
                    x={d.x - cplHalf - 6}
                    y={d.y + Math.min(0, gapDir) - 10}
                    width={cplHalf * 2 + 12}
                    height={Math.abs(gapDir) + 20}
                  />
                  <line
                    className="device-line"
                    x1={d.x - cplHalf}
                    y1={d.y}
                    x2={d.x + cplHalf}
                    y2={d.y}
                  />
                  <line
                    className="device-line"
                    x1={d.x - cplHalf}
                    y1={d.y + gapDir}
                    x2={d.x + cplHalf}
                    y2={d.y + gapDir}
                  />
                  <text className="caption" x={d.x} y={d.y + 20} textAnchor="middle">
                    coupler
                  </text>
                </>
              ) : null}

              {d.type === "ring" ? (
                <>
                  {selected ? <circle className="halo" cx={d.x} cy={d.y} r={r + 8} /> : null}
                  <circle className="hit-fill" cx={d.x} cy={d.y} r={r + 10} />
                  <circle className="device-line" cx={d.x} cy={d.y} r={r} />
                  <text className="caption" x={d.x} y={d.y + r + 16} textAnchor="middle">
                    ring
                  </text>
                </>
              ) : null}

              {d.type === "detector" ? (
                <>
                  {selected ? (
                    <rect className="halo" x={d.x - 18} y={d.y - 18} width={40} height={36} rx={4} />
                  ) : null}
                  <rect className="hit-fill" x={d.x - 16} y={d.y - 16} width={36} height={32} />
                  <polygon
                    className="detector"
                    points={`${d.x - 12},${d.y - 12} ${d.x + 14},${d.y} ${d.x - 12},${d.y + 12}`}
                  />
                  <line
                    className="device-line"
                    x1={d.x + 14}
                    y1={d.y - 10}
                    x2={d.x + 14}
                    y2={d.y + 10}
                  />
                  <text className="caption" x={d.x} y={d.y + 28} textAnchor="middle">
                    detector
                  </text>
                </>
              ) : null}
            </g>
          );
        })}
      </svg>
      <p className="board-hint">Parts sit in a typical layout. Drag only to tweak.</p>
    </div>
  );
}

function CrossSection({
  widthNm,
  heightNm,
  nClad,
  field,
  onWidthChange,
  onHeightChange,
}: {
  widthNm: number;
  heightNm: number;
  nClad: number;
  field?: FieldMap;
  onWidthChange: (widthNm: number) => void;
  onHeightChange: (heightNm: number) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<"width" | "height" | null>(null);
  const w = Math.max(80, widthNm * 0.4);
  const h = Math.max(24, heightNm * 0.4);
  const clad = nClad === 1 ? "air" : nClad > 1.4 ? "oxide" : "water";

  function apply(e: PointerEvent<SVGElement>, mode: "width" | "height") {
    const svg = svgRef.current;
    if (!svg) return;
    const p = toSvgPoint(e, svg);
    if (!p) return;
    if (mode === "width") {
      onWidthChange(Math.min(800, Math.max(250, Math.round((Math.abs(p.x) * 2) / 0.4 / 10) * 10)));
    } else {
      onHeightChange(Math.min(340, Math.max(150, Math.round(Math.abs(p.y) / 0.4 / 10) * 10)));
    }
  }

  return (
    <svg
      ref={svgRef}
      className="schematic section"
      viewBox="-220 -160 440 280"
      aria-label="Waveguide cross-section"
    >
      <rect className="clad" x={-200} y={-150} width={400} height={150} />
      {field?.x_nm && field.y_nm ? (
        <image
          href={fieldToDataUrl(field)}
          x={field.x_nm[0] * 0.4}
          y={-(field.y_nm[field.y_nm.length - 1] ?? 0) * 0.4}
          width={(field.x_nm[field.x_nm.length - 1] - field.x_nm[0]) * 0.4}
          height={(field.y_nm[field.y_nm.length - 1] - field.y_nm[0]) * 0.4}
          preserveAspectRatio="none"
          opacity={0.95}
          pointerEvents="none"
        />
      ) : null}
      <rect className={`core ${field ? "field-on" : ""}`} x={-w / 2} y={-h} width={w} height={h} />
      <rect className="box" x={-200} y={0} width={400} height={36} />
      <text x={0} y={-h - 16} textAnchor="middle">
        {clad} n = {nClad}
      </text>
      <text x={0} y={h / -2 + 4} textAnchor="middle">
        Si {widthNm} × {heightNm} nm
      </text>
      <text x={0} y={24} textAnchor="middle">
        BOX
      </text>
      <rect
        className="handle-box"
        x={w / 2 - 6}
        y={-h / 2 - 6}
        width={12}
        height={12}
        onPointerDown={(e) => {
          drag.current = "width";
          e.currentTarget.setPointerCapture(e.pointerId);
          apply(e, "width");
        }}
        onPointerMove={(e) => {
          if (drag.current !== "width" || !e.currentTarget.hasPointerCapture(e.pointerId)) return;
          apply(e, "width");
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
      />
      <rect
        className="handle-box"
        x={-6}
        y={-h - 6}
        width={12}
        height={12}
        onPointerDown={(e) => {
          drag.current = "height";
          e.currentTarget.setPointerCapture(e.pointerId);
          apply(e, "height");
        }}
        onPointerMove={(e) => {
          if (drag.current !== "height" || !e.currentTarget.hasPointerCapture(e.pointerId)) return;
          apply(e, "height");
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
      />
    </svg>
  );
}

function Bar({
  side,
  title,
  open,
  onToggle,
  children,
  stacked = false,
  grow = false,
}: {
  side: "left" | "right";
  title: string;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
  stacked?: boolean;
  grow?: boolean;
}) {
  return (
    <aside
      className={`bar ${side} ${open ? "open" : "closed"} ${stacked ? "stacked" : ""} ${grow ? "grow" : ""}`}
    >
      <button type="button" className="bar-toggle" onClick={onToggle} aria-expanded={open}>
        <span>{title}</span>
        <span className={`bar-chevron ${open ? "open" : ""}`} aria-hidden="true">
          <svg viewBox="0 0 16 16" width="16" height="16">
            <path
              d="M6 3.5 11 8 6 12.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </button>
      {open ? <div className="bar-body">{children}</div> : null}
    </aside>
  );
}

function ResultFold({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="result-fold">
      <button
        type="button"
        className="result-fold-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span>{title}</span>
        <span className={`bar-chevron ${open ? "open" : ""}`} aria-hidden="true">
          <svg viewBox="0 0 16 16" width="16" height="16">
            <path
              d="M6 3.5 11 8 6 12.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </button>
      {open ? <div className="result-fold-body">{children}</div> : null}
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  step = "any",
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: string;
  min?: number;
  max?: number;
}) {
  return (
    <label>
      {label}
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

function topologyObjectiveNote(report: TopologyResult): string {
  const h = report.history;
  const first = h[0];
  const last = h[h.length - 1];
  const target = report.kappa_target;
  const start = first.objective;
  const end = last.objective;
  const ratio = end / (Math.abs(start) > 1e-12 ? start : 1e-12);
  let trend: string;
  if (end < 1e-4) {
    trend = `ended near zero (${fmt(end, 4)}), so the worst blueprint is on the κ target`;
  } else if (ratio < 0.7) {
    trend = `fell from ${fmt(start, 3)} to ${fmt(end, 3)}, so the worst blueprint moved closer to the target`;
  } else if (ratio > 1.2) {
    trend = `rose from ${fmt(start, 3)} to ${fmt(end, 3)}; the worst-case split did not improve`;
  } else {
    trend = `stayed near ${fmt(end, 3)}; the worst-case error barely moved`;
  }
  const counts = { dilated: 0, intermediate: 0, eroded: 0 };
  for (const step of h) {
    if (step.worst) counts[step.worst] += 1;
  }
  const counted = (Object.entries(counts) as [keyof typeof counts, number][]).filter(
    ([, n]) => n > 0
  );
  const who =
    counted.length > 0
      ? ` The adjoint used ${counted.map(([name, n]) => `${name} on ${n} step${n === 1 ? "" : "s"}`).join(", ")}.`
      : "";
  const tgt = target != null ? ` Target κ is ${fmt(target, 2)}.` : "";
  const lastWorst = last.worst ? ` Last update was the ${last.worst} geometry.` : "";
  return `Each point is the largest (T_drop − κ)² among dilated / intermediate / eroded, plus a small grayscale penalty on the intended layout.${tgt} The curve ${trend}.${who}${lastWorst}`;
}

function topologyTDropNote(
  report: TopologyResult,
  etas: { dilated: number; intermediate: number; eroded: number }
): string {
  const last = report.history[report.history.length - 1];
  const d = last.t_drop_dilated;
  const m = last.t_drop_intermediate;
  const e = last.t_drop_eroded;
  if (d == null || m == null || e == null) return "";
  const target = report.kappa_target ?? 0.5;
  const rows: [string, number][] = [
    ["dilated", d],
    ["intermediate", m],
    ["eroded", e],
  ];
  const farthest = rows.reduce((a, b) =>
    Math.abs(b[1] - target) > Math.abs(a[1] - target) ? b : a
  );
  const spread = Math.abs(d - e);
  const final = report.robust;
  const finalBit =
    final?.dilated && final?.eroded
      ? ` On the finished layout, T_drop is ${fmt(final.dilated.t_drop, 3)} dilated and ${fmt(final.eroded.t_drop, 3)} eroded${final.worst ? ` (${final.worst} is worst)` : ""}.`
      : "";
  const spreadBit =
    spread < 0.05
      ? ` Dilated and eroded stay within ${fmt(spread, 3)}, so the split is fairly stable to over/under-etch.`
      : ` Dilated and eroded differ by ${fmt(spread, 3)}; bias still moves the split.`;
  return `The three traces are the same filtered density with η = ${fmt(etas.dilated, 2)} (features grow), ${fmt(etas.intermediate, 2)} (intended chip), and ${fmt(etas.eroded, 2)} (features shrink). Target κ is ${fmt(target, 2)}. Last step: dilated ${fmt(d, 3)}, intermediate ${fmt(m, 3)}, eroded ${fmt(e, 3)}. ${farthest[0]} is farthest from the target (|Δκ| = ${fmt(Math.abs(farthest[1] - target), 3)}).${spreadBit}${finalBit}`;
}

function topologyDensityNote(
  report: TopologyResult,
  etas: { dilated: number; intermediate: number; eroded: number }
): string {
  const gray = report.grayscale;
  const fillPct = report.drc.fill * 100;
  const grayBit =
    gray < 0.08
      ? `Little leftover gray (${fmt(gray, 3)}), so most pixels are binary Si or clad.`
      : `Grayscale is still ${fmt(gray, 3)}; some pixels are neither fully Si nor clad.`;
  const mfs = report.drc.mfs_ok
    ? `min feature ${fmt(report.drc.min_feature_nm, 0)} nm meets the ${fmt(report.drc.mfs_nm, 0)} nm DUV rule`
    : `min feature ${fmt(report.drc.min_feature_nm, 0)} nm is below the ${fmt(report.drc.mfs_nm, 0)} nm DUV MFS`;
  const gap = report.drc.gap_ok
    ? `min gap ${fmt(report.drc.min_gap_meas_nm, 0)} nm is OK`
    : `min gap ${fmt(report.drc.min_gap_meas_nm, 0)} nm is below the ${fmt(report.drc.min_gap_nm, 0)} nm rule`;
  const t =
    report.t_drop != null
      ? ` Intermediate T_drop is ${fmt(report.t_drop, 3)}.`
      : "";
  return `Dark is silicon after Helmholtz filtering, tanh projection at η = ${fmt(etas.intermediate, 2)} (the layout you would write), and litho/etch bias. Fill is ${fmt(fillPct, 0)}%. ${grayBit} ${mfs}; ${gap}.${t} ${report.polygon_count} polygon${report.polygon_count === 1 ? "" : "s"} in the GDS.`;
}

function TopologyPanel({
  report,
  busy,
  onRun,
}: {
  report: TopologyResult | null;
  busy: boolean;
  onRun: () => void;
}) {
  const etas = report?.etas ?? { dilated: 0.3, intermediate: 0.5, eroded: 0.7 };
  return (
    <>
      <p className="muted">
        Density TO of the coupler: Helmholtz filter and tanh projection for DUV min
        feature size. Each step uses the worst κ error of dilated / intermediate /
        eroded (η = {fmt(etas.dilated, 2)} / {fmt(etas.intermediate, 2)} /{" "}
        {fmt(etas.eroded, 2)}; Piggott / Wang), then litho/etch bias and GDS. Same 2.5D
        strip as FDTD — not a 3D ring solve.
      </p>
      <button type="button" onClick={onRun} disabled={busy}>
        {busy ? "Running topology…" : "Run topology"}
      </button>
      {report ? (
        <>
          <div className="metrics">
            <Metric label="Δx" value={`${fmt(report.dx_nm, 0)} nm`} />
            <Metric label="β" value={fmt(report.beta, 1)} />
            <Metric
              label="η dilated / mid / eroded"
              value={`${fmt(etas.dilated, 2)} / ${fmt(etas.intermediate, 2)} / ${fmt(etas.eroded, 2)}`}
            />
            <Metric label={`T_drop (η=${fmt(etas.intermediate, 2)})`} value={fmt(report.t_drop, 3)} />
            <Metric label="Gray" value={fmt(report.grayscale, 3)} />
            <Metric
              label="MFS"
              value={report.drc.mfs_ok ? `${fmt(report.drc.min_feature_nm, 0)} nm` : "fail"}
            />
            <Metric
              label="Gap"
              value={report.drc.gap_ok ? `${fmt(report.drc.min_gap_meas_nm, 0)} nm` : "fail"}
            />
            <Metric label="Polygons" value={report.polygon_count} />
          </div>
          <p className="muted">
            {report.filter}. {report.projection}. {report.vectorizer}.{" "}
            {report.fabrication}.
          </p>
          {report.robust?.dilated || report.robust?.eroded || report.robust?.intermediate ? (
            <div className="metrics">
              {report.robust?.dilated ? (
                <Metric
                  label={`Dilated T_drop (η=${fmt(etas.dilated, 2)})`}
                  value={fmt(report.robust.dilated.t_drop, 3)}
                />
              ) : null}
              {report.robust?.eroded ? (
                <Metric
                  label={`Eroded T_drop (η=${fmt(etas.eroded, 2)})`}
                  value={fmt(report.robust.eroded.t_drop, 3)}
                />
              ) : null}
              {report.robust?.t_drop_worst != null ? (
                <Metric
                  label={`Worst T_drop (${report.robust.worst ?? "—"})`}
                  value={fmt(report.robust.t_drop_worst, 3)}
                />
              ) : null}
            </div>
          ) : null}
          {report.gds_b64 ? (
            <button type="button" onClick={() => downloadGds(report.gds_b64)}>
              Download GDS ({report.gds_bytes} B)
            </button>
          ) : null}
          {report.history.length > 1 ? (
            <>
              <LinePlot
                title="Worst-case TO objective"
                xLabel="Step"
                yLabel="max (T_drop − target)²"
                x={report.history.map((h) => h.step)}
                series={[
                  {
                    name: "worst κ error",
                    y: report.history.map((h) => h.objective),
                    kind: "through",
                  },
                ]}
                yAuto
                xDigits={0}
                note={topologyObjectiveNote(report)}
                noteLabel="Objective"
              />
              {report.history.every(
                (h) =>
                  h.t_drop_dilated != null &&
                  h.t_drop_intermediate != null &&
                  h.t_drop_eroded != null
              ) ? (
                <LinePlot
                  title="T_drop dilated / intermediate / eroded"
                  xLabel="Step"
                  yLabel="T_drop"
                  x={report.history.map((h) => h.step)}
                  series={[
                    {
                      name: "dilated",
                      y: report.history.map((h) => h.t_drop_dilated as number),
                      kind: "drop",
                    },
                    {
                      name: "intermediate",
                      y: report.history.map((h) => h.t_drop_intermediate as number),
                      kind: "through",
                    },
                    {
                      name: "eroded",
                      y: report.history.map((h) => h.t_drop_eroded as number),
                      kind: "mid",
                    },
                  ]}
                  xDigits={0}
                  note={topologyTDropNote(report, etas)}
                  noteLabel="T_drop"
                />
              ) : null}
            </>
          ) : null}
          {report.field ? (
            <Heatmap
              title="Projected density  (dark = Si)"
              xLabel="y (μm)"
              yLabel="x (μm)"
              field={report.field}
              note={topologyDensityNote(report, etas)}
              noteLabel="Projected density"
            />
          ) : null}
        </>
      ) : null}
    </>
  );
}

function FdtdPanel({
  report,
  compactNeff,
  busy,
  onRun,
}: {
  report: FdtdReport;
  compactNeff: number;
  busy: boolean;
  onRun: () => void;
}) {
  const g = report.geometry;
  const mesh = report.mesh;
  const mat = report.materials;
  const bc = report.boundaries;
  const fde = report.mode;
  const run = report.run;
  return (
    <>
      <p className="muted">
        Design numbers use fitted compact models. This panel is the FDTD setup for that
        geometry. Run 2D FDTD for a scalar FDE n_eff and a coarse in-plane coupler split —
        not a 3D ring solve.
      </p>
      <div className="metrics">
        <Metric label="Primitives" value={g.primitives.length} />
        <Metric label="Yee cells" value={mesh.cells} />
        <Metric label="Δx" value={`${fmt(mesh.dx_nm, 0)} nm`} />
        <Metric label="PML" value={`${bc.npml} cells`} />
        <Metric
          label="FDE n_eff"
          value={fde?.n_eff != null ? fmt(fde.n_eff) : "—"}
        />
        <Metric label="Fitted n_eff" value={fmt(compactNeff)} />
      </div>
      <p className="muted">
        {g.primitives.map((p) => p.role || p.kind).join(", ") || "empty"}.{" "}
        {mesh.subpixel}. {mat.dispersion}. {bc.absorbing}.
      </p>
      <button type="button" onClick={onRun} disabled={busy}>
        {busy ? "Running 2D FDTD…" : "Run 2D FDTD"}
      </button>
      {run?.error ? <p className="muted">{run.error}</p> : null}
      {run && !run.error ? (
        <>
          <p className="muted">
            Port split is |E|² at the end of the 2D coupler, not the design κ.
          </p>
          <div className="metrics">
            <Metric label="Steps" value={run.steps ?? "—"} />
            <Metric label="P_through" value={fmt(run.t_through, 3)} />
            <Metric label="P_drop" value={fmt(run.t_drop, 3)} />
            <Metric label="Split" value={fmt(run.kappa, 3)} />
          </div>
        </>
      ) : null}
      {run?.field ? (
        <Heatmap
          title="2D FDTD |Ez| snapshot"
          xLabel="y (μm)"
          yLabel="x (μm)"
          field={run.field}
        />
      ) : fde?.field ? (
        <Heatmap
          title="FDE eigenmode |ψ|²"
          xLabel="x (nm)"
          yLabel="y (nm)"
          field={fde.field}
        />
      ) : null}
    </>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PlotNoteCard({ label, note }: { label: string; note?: string }) {
  if (!note) return null;
  const bits = note
    .split(/(?<=\.)\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
  return (
    <div className="metric plot-note-card">
      <span>{label}</span>
      {bits.length > 1 ? (
        <ul>
          {bits.map((bit) => (
            <li key={bit}>{bit}</li>
          ))}
        </ul>
      ) : (
        <strong>{note}</strong>
      )}
    </div>
  );
}

function downloadGds(b64: string) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const url = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = "coupler_to.gds";
  a.click();
  URL.revokeObjectURL(url);
}

function toSvgPoint(e: PointerEvent<SVGElement>, svg: SVGSVGElement) {
  const ctm = svg.getScreenCTM();
  if (!ctm) return null;
  const pt = svg.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  return pt.matrixTransform(ctm.inverse());
}

function SParamTable({ s }: { s?: SParams }) {
  if (!s) return null;
  return (
    <ResultFold title="S-parameters">
      <div className="metrics">
        <Metric label="|S11|" value={fmt(s.s11, 4)} />
        <Metric label="|S21|" value={fmt(s.s21, 4)} />
        <Metric label="|S31|" value={fmt(s.s31, 4)} />
        <Metric label="|S41|" value={fmt(s.s41, 4)} />
        <Metric label="|S21| (dB)" value={fmt(s.s21_db, 2)} />
        <Metric label="|S31| (dB)" value={fmt(s.s31_db, 2)} />
      </div>
    </ResultFold>
  );
}

function jetRgb(v: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, v));
  let r = 0;
  let g = 0;
  let b = 0;
  if (t < 0.125) {
    b = 0.55 + 3.6 * t;
  } else if (t < 0.375) {
    g = (t - 0.125) * 4;
    b = 1;
  } else if (t < 0.625) {
    r = (t - 0.375) * 4;
    g = 1;
    b = 1 - (t - 0.375) * 4;
  } else if (t < 0.875) {
    r = 1;
    g = 1 - (t - 0.625) * 4;
  } else {
    r = 1;
    g = 0;
  }
  return [Math.round(255 * r), Math.round(255 * g), Math.round(255 * b)];
}

function densityRgb(v: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, v));
  return [
    Math.round(236 + (44 - 236) * t),
    Math.round(236 + (44 - 236) * t),
    Math.round(230 + (42 - 230) * t),
  ];
}

function fieldToDataUrl(field: FieldMap) {
  const z = field.intensity;
  const ny = z.length;
  const nx = z[0]?.length ?? 0;
  if (!nx || !ny) return "";
  const canvas = document.createElement("canvas");
  canvas.width = nx;
  canvas.height = ny;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const img = ctx.createImageData(nx, ny);
  const rgb = field.colormap === "density" ? densityRgb : jetRgb;
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const [r, g, b] = rgb(z[iy][ix]);
      const row = ny - 1 - iy;
      const i = (row * nx + ix) * 4;
      img.data[i] = r;
      img.data[i + 1] = g;
      img.data[i + 2] = b;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

function Heatmap({
  title,
  xLabel,
  yLabel,
  field,
  note,
  noteLabel = "Reading",
}: {
  title: string;
  xLabel: string;
  yLabel: string;
  field: FieldMap;
  note?: string;
  noteLabel?: string;
}) {
  const x = field.x_nm ?? field.x_um ?? [];
  const y = field.y_nm ?? field.z_um ?? [];
  const z = field.intensity;
  const outlines = field.guides ?? (field.core ? [field.core] : []);
  const url = fieldToDataUrl(field);

  if (!x.length || !y.length || !z.length) return null;
  const xmin = x[0];
  const xmax = x[x.length - 1];
  const ymin = y[0];
  const ymax = y[y.length - 1];
  const aspect = (xmax - xmin) / (ymax - ymin || 1);
  const maxW = 270;
  const maxH = field.colormap === "density" || field.layout === "strip" ? 120 : 170;
  const tickD = Math.max(Math.abs(xmax - xmin), Math.abs(ymax - ymin)) >= 20 ? 0 : 2;
  const plotW = aspect > maxW / maxH ? maxW : maxH * aspect;
  const plotH = aspect > maxW / maxH ? maxW / aspect : maxH;
  const barW = 12;
  const pad = { l: 44, r: 48, t: 8, b: 36 };
  const svgW = pad.l + plotW + pad.r;
  const svgH = pad.t + plotH + pad.b;

  function px(v: number) {
    return pad.l + ((v - xmin) / (xmax - xmin || 1)) * plotW;
  }
  function py(v: number) {
    return pad.t + ((ymax - v) / (ymax - ymin || 1)) * plotH;
  }

  return (
    <div className="plot-block">
      <p className="bar-section">{title}</p>
      <svg className="plot field-axes" viewBox={`0 0 ${svgW} ${svgH}`}>
        <image
          href={url}
          x={pad.l}
          y={pad.t}
          width={plotW}
          height={plotH}
          preserveAspectRatio="none"
        />
        <rect x={pad.l} y={pad.t} width={plotW} height={plotH} fill="none" stroke="#ccc" />
        {outlines.map((o, i) => (
          <rect
            key={i}
            className="field-core"
            x={px(o.x0)}
            y={py(o.y0 + o.height)}
            width={px(o.x0 + o.width) - px(o.x0)}
            height={py(o.y0) - py(o.y0 + o.height)}
          />
        ))}
        {field.polygons?.map((poly, i) => (
          <polygon
            key={`p${i}`}
            className="field-core"
            points={poly.map((p) => `${px(p.x)},${py(p.y)}`).join(" ")}
          />
        ))}
        {field.core ? (
          <line className="field-core" x1={pad.l} y1={py(0)} x2={pad.l + plotW} y2={py(0)} />
        ) : null}
        {Array.from({ length: 48 }, (_, i) => {
          const t = i / 47;
          const [r, g, b] = field.colormap === "density" ? densityRgb(t) : jetRgb(t);
          return (
            <rect
              key={i}
              x={pad.l + plotW + 10}
              y={pad.t + plotH - ((i + 1) * plotH) / 48}
              width={barW}
              height={plotH / 48 + 0.5}
              fill={`rgb(${r},${g},${b})`}
            />
          );
        })}
        <text x={pad.l + plotW + 10 + barW + 4} y={pad.t + 10}>
          1
        </text>
        <text x={pad.l + plotW + 10 + barW + 4} y={pad.t + plotH}>
          0
        </text>
        <text x={pad.l} y={svgH - 8}>
          {xmin.toFixed(tickD)}
        </text>
        <text x={pad.l + plotW} y={svgH - 8} textAnchor="end">
          {xmax.toFixed(tickD)} {xLabel}
        </text>
        <text x={8} y={pad.t + 10}>
          {ymax.toFixed(tickD)}
        </text>
        <text x={8} y={pad.t + plotH}>
          {ymin.toFixed(tickD)}
        </text>
        <text x={12} y={pad.t + plotH / 2} transform={`rotate(-90 12 ${pad.t + plotH / 2})`}>
          {yLabel}
        </text>
      </svg>
      <PlotNoteCard label={noteLabel} note={note} />
    </div>
  );
}

function LinePlot({
  title,
  xLabel,
  yLabel,
  x,
  series,
  db = false,
  markX,
  xDigits,
  yAuto = false,
  note,
  noteLabel = "Reading",
}: {
  title: string;
  xLabel: string;
  yLabel: string;
  x: number[];
  series: { name: string; y: number[]; kind: "through" | "drop" | "mid" }[];
  db?: boolean;
  markX?: number;
  xDigits?: number;
  yAuto?: boolean;
  note?: string;
  noteLabel?: string;
}) {
  if (!x.length || !series[0]?.y.length) return null;
  const width = 640;
  const height = 240;
  const pad = { l: 52, r: 16, t: 12, b: 36 };
  const x0 = pad.l;
  const y0 = pad.t;
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const xmin = x[0];
  const xmax = x[x.length - 1];
  const mapped = series.map((s) => ({
    ...s,
    y: db ? s.y.map((v) => 10 * Math.log10(Math.max(v, 1e-8))) : s.y,
  }));
  let ymin = yAuto ? mapped[0].y[0] : db ? -40 : 0;
  let ymax = yAuto ? mapped[0].y[0] : db ? 0 : 1;
  for (const s of mapped) {
    for (const v of s.y) {
      if (v < ymin) ymin = v;
      if (v > ymax) ymax = v;
    }
  }
  if (yAuto) {
    const padY = (ymax - ymin) * 0.12 || 0.01;
    ymin -= padY;
    ymax += padY;
  } else if (!db) {
    ymin = 0;
    ymax = Math.max(1, ymax);
  } else {
    ymin = Math.min(-40, ymin);
    ymax = 0;
  }
  const xd = xDigits ?? (db ? 1 : 1);
  const yd = yAuto ? 2 : db ? 0 : 1;

  function px(v: number) {
    return x0 + ((v - xmin) / (xmax - xmin || 1)) * innerW;
  }
  function py(v: number) {
    return y0 + ((ymax - v) / (ymax - ymin || 1)) * innerH;
  }
  function path(ys: number[]) {
    return ys.map((v, i) => `${i ? "L" : "M"} ${px(x[i])} ${py(v)}`).join(" ");
  }

  return (
    <div className="plot-block">
      <p className="bar-section">{title}</p>
      <svg className="plot" viewBox={`0 0 ${width} ${height}`}>
        <line x1={x0} y1={y0} x2={x0} y2={y0 + innerH} />
        <line x1={x0} y1={y0 + innerH} x2={x0 + innerW} y2={y0 + innerH} />
        {mapped.map((s) => (
          <path key={s.name} d={path(s.y)} className={s.kind} />
        ))}
        {markX != null && markX >= xmin && markX <= xmax ? (
          <line className="guide" x1={px(markX)} y1={y0} x2={px(markX)} y2={y0 + innerH} />
        ) : null}
        <text x={x0} y={height - 6}>
          {xmin.toFixed(xd)}
        </text>
        <text x={x0 + innerW} y={height - 6} textAnchor="end">
          {xmax.toFixed(xd)} {xLabel}
        </text>
        <text x={8} y={y0 + 10}>
          {ymax.toFixed(yd)}
        </text>
        <text x={8} y={y0 + innerH}>
          {ymin.toFixed(yd)}
        </text>
        <text x={14} y={y0 + innerH / 2} transform={`rotate(-90 14 ${y0 + innerH / 2})`}>
          {yLabel}
        </text>
      </svg>
      <p className="legend">
        {mapped.map((s) => (
          <span key={s.name} className={s.kind}>
            {s.name}
          </span>
        ))}
      </p>
      <PlotNoteCard label={noteLabel} note={note} />
    </div>
  );
}
