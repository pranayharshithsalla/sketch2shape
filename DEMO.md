# Sketch2Shape: 3-minute demo script

**Setup (before you go on stage):** run `python app.py` (or `.\run.ps1` on Windows) and open
http://127.0.0.1:7860. Keep your phone on the same Wi-Fi as the laptop. Leave `samples/bat/` open in
a file window as a backup in case the network or quota fails.

| Time | Say | Do |
|---|---|---|
| **0:00 – 0:20** | "Sketch2Shape turns a sketch, a photo or a sentence into a real, printable 3D model. It's one evolving design, not three separate tools." | Show the Studio: top bar with three modes, grid sketchpad, 3D preview, metrics. |
| **0:20 – 0:45** | "Start with words only: a quick start." | Pick **💬 Explain what you need** and type `a wooden cricket bat with a rubber grip`. Click **Create 3D Shape**. v1 appears (≈10–15 s). Rotate it with the ⟲/⟳ buttons. |
| **0:45 – 1:05** | "Here's what AI can't do: ask any image model to write *BRAND NAME* on the blade and you get gibberish. Text is a known weakness of generative models." | (Optional) Show a garbled text candidate from the Advanced tab, or just say it. |
| **1:05 – 1:30** | "Now refine it with a reference photo. I choose what it controls: *Style & colour only*, so the shape stays locked." | Pick **📷 Upload an image**, drop a photo of a bat or any wood texture, set **This input controls → Style & colour only**, then **Update 3D Shape**. v2 appears in the **version timeline**. |
| **1:30 – 1:55** | "For sketches, the app generates four candidates and ranks them: 50% sketch match, 30% 3D-readiness, 20% prompt match. Only the winner goes to the expensive 3D step." | Pick **✏️ Draw rough sketch**, draw a bat outline, click **🎲 Generate 4 candidates**. Point at the **★ BEST** card and its score breakdown. (Tick **Auto** to send it straight to 3D.) |
| **1:55 – 2:20** | "Now the text. We draw it ourselves and bake it into the model's texture, so it's crisp, lives inside the .glb, and survives later edits." | Open **🔤 Text & logo**, type `BRAND NAME`, font *Heavy*, rotation 90, size 0.7, then **Bake onto model**. Tick **Live move** and drag the position slider to slide it along the blade. |
| **2:20 – 2:35** | "Every version is scored: Accuracy compares the model's silhouette to the sketch. If it's too low we automatically re-plan, up to twice. Print-ready means every part is watertight." | Point at **Accuracy**, **Print ready**, **Parts**, **Build time**. Open **✅ Validation report**: vertices, faces, watertight, texture present, file size, re-open check. |
| **2:35 – 2:55** | "And it's real 3D: scan this." | Open **📱 View in AR**, scan the QR with your phone, tap **View in your room**, and place the bat on the table. |
| **2:55 – 3:00** | "Download as GLB, OBJ or STL, revert any version with one tap. It works offline too." | Click **Download 3D Model (.GLB)** and **↩ Revert to selected** on v1. |

**If something fails live:**
- Planner quota hit → the reply says it fell back to the local engines; continue with a sketch.
- Wi-Fi blocks the phone → open the AR link on the laptop instead (desktop 3D view), or show `samples/bat/turntable.gif`.
- Show the cached results: `samples/<name>/` holds the input, the 4 ranked candidates, `model.glb`, a turntable and `result.json`.
