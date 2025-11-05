### Developer Templates for PolSAR Processing

This folder contains template scripts for core processing modules in the `polsartools` package. These examples demonstrate how to import sensor data, convert it into polarimetric matrices,  processing the data in different modes to get the required output parameter/s.

---

### Template Overview

### 1. `import_sensor.py`
This script demonstrates how to read dual-pol/Full-pol sensor data from a supported format and extract either Single look complex (SLC) elements (S2,Sxy) or multilooked PolSAR matrices (C4/T4/C3/T3/C2/T2).

### 2. `function_fp.py`
This script demonstrates how to process a full-pol matrix elements to get required output parameter/s. 

### 3. `function_dp.py`
This script demonstrates how to process a dual-pol matrix elements to get required output parameter/s.

### 4. `function_cp.py`
This script demonstrates how to process a compact-pol matrix elements to get required output parameter/s.
