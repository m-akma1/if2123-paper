# SVD-Stabilized Perspective-n-Point Estimation for Vision-Based Relative Localization in UAV Formation Systems

This repository contains a minimal Python experiment to compare various methods for solving the Perspective-n-Point (PnP) problem. The baseline method uses Efficient Pnp (EPnP) and Iterative PnP, while the proposed method modifiying the EPnP by using an iterative approach while stabilizing the result with SVD. 

## Getting Started

1. Clone the repository
```bash
git clone https://github.com/m-akma1/if2123-paper.git
cd if2123-paper
```

2. Setup Python Virtual Environment  

* On MacOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```
* On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the experiment and plot the result
> [!WARNING]  
> The current experiment result will be overwritten  
> Please move/copy the original result if you want to   save it

```bash
python scripts/run_sweep.py
python scripts/plot_result.py
```

## Credits

Created by Muhammad Akmal (13524099)  
For IF2123 Linear and Geometric Algebra Paper Assignment  
Semester 1 Academic Year 2025/2026 ITB  

