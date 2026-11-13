# BayCAST mirror

A small public copy of the newest surface currents from the Texas Water Development Board (TWDB) BayCAST model,
in a form a web page can read.

- Source: https://midgewater.twdb.texas.gov/bays_estuaries/baycast/ (daily files `hydro_2D_surf_YYYYMMDD.nc`)
- A scheduled job (`.github/workflows/mirror.yml`) runs `mirror.py` every 3 hours and publishes the result on the `gh-pages` branch.
- Kept: the last 60 days, plus the forecast days TWDB publishes. Only the Texas side of the mesh (west of longitude -93.3).
- `index.json` lists the days. `days/YYYYMMDD.bin` holds u then v as int8 (value x 0.02 = m/s). `mesh_<key>.bin` holds the mesh.

This is not an official TWDB product. TWDB says BayCAST is for research and education, not for real-time decisions.
The data belong to TWDB (file attribute `license`: "Freely Distributed").
