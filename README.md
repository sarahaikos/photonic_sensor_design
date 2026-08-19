# Photonic sensor design

Waveguide indices, overlap, and directional-coupler κ come from **compact models fitted to 3D FDTD / vectorial FDE** of 220 nm SOI. Results include FDTD-style **S-parameters** (`S11`, `S21`, `S31`, `S41`), **T_through** / **T_drop**, and **extracted** resonator parameters (λ0, FWHM, Q_loaded, Q_i, Q_c, extinction, κ).

Stack: Python (FastAPI) + React (Vite + TypeScript).

## Run

Terminal 1 — Python API:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Terminal 2 — React UI:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Inputs

| Input | Meaning |
| --- | --- |
| Configuration | All-pass (one bus) or add-drop (two buses) |
| Radius | Ring radius |
| Wavelength | Design wavelength |
| n_eff, n_g | Effective and group index |
| κ | Power coupling to the bus |
| Waveguide loss | Propagation loss |
| dn_eff / dn | How much n_eff changes with cladding index (overlap) |

## Outputs

- Nearest resonance and mode order
- Free spectral range, Q, FWHM, finesse
- Under / critical / over coupling
- Bulk sensitivity `λ · (dn_eff/dn) / n_g`
- Analyte sweep: resonance shift and T at the laser wavelength vs cladding index
- Tune ring radius so a resonance sits on the laser λ
- Critical coupling: set coupler gap (or length) so round-trip loss matches κ
- Rough limit of detection `λ / (10 · Q · S)`
- FDTD setup for the chip (geometry, Yee mesh, materials, PML, sources, monitors)
- Optional **Run 2D FDTD**: scalar FDE n_eff plus a coarse in-plane coupler port split

Live design numbers (Q, κ, sweep, S-params) come from FDTD-fitted compact models. The 2D run is not a 3D ring simulation.
