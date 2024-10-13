# LOT STREAMING IN JOB SHOP SCHEDULING

## Main features
- 4 MILP modelled in **Gurobi** (needed a license to solve them, free for students). **All possible combinations to handle sequence independent/dependent setup times and shift constraints**. Models are
  - **basic**: sequence independent setup times and no shifts
  - **shift constraints**: sequenc independent setup times and shifts
  - **sequence dependent setup times**: no shifts
  - **sequence dependent setup times and shift constraints**

- **Plott of job shop gantt chart using plotly.express.timeline** (instead of plotly.figure_factory)
  
| ![v5_m3_j3_u3_s5_d200_setup_no_pmtn](https://github.com/user-attachments/assets/557e6060-2197-4fc2-b87e-bf81ca3a5fc3) |
|:--:| 
| *Lot Streaming Job Shop Scheduling through MILP solving* |

- **Plot of the evolution of the solution** (MIP Gap and objective function)
  
| ![v5_m4_j4_u3_s12_d300_setup_no_pmtn_MAKESPAN_EVOLUTION](https://github.com/user-attachments/assets/645adab1-9472-4be2-9104-16bda65cf048) |
|:--:| 
| *Solution evolution: MIP Gap and Objective Function (Makespan)* |

- **Generation of random parameters for job shop problems**
- PEP8 (guidestyle) compliance 

## Mixed Integer Linear Programming Models
All them can be found in the "MILP_LSJSP.pdf file. They are in spanish, sorry for that. Feel free to do the translation and collaborate with this project!

## Secondary contents
- drafts of extremely simple models in pyomo (equal size lot streaming)
- drafts of gurobi models (used when learning and researching this problem)

