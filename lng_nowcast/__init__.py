"""lng_nowcast — reconstruct NW-European LNG terminal send-out from free public data.

Layers (built up over the project):
  1. Ingestion: ENTSOG transparency (keyless), GIE ALSI (free key), AIS (later).
  2. Revision logging: twice-daily snapshots of what each source *currently* claims
     about recent gas days, so the Estimated->Confirmed revision process can be
     measured empirically. This only accumulates in real time — it runs from day one.
  3. Physical inversion: volumes/draughts -> energy, with propagated error bars.
  4. State-space filter: posterior over true send-out ahead of confirmation.
"""

__version__ = "0.1.0"
