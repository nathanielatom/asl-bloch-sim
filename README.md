# Arterial Spin Labelling (ASL) Bloch Simulations

## Pseudo-continuous ASL for different blood flow trajectories

### Linear Flow

Linear or constant velocity blood flow is modelled in `Bloch PCASL Full Relaxation.ipynb`, `Bloch PCASL Max Labelling Efficiency Observed.ipynb`, and technically `Bloch PCASL No Flow.ipynb`. An example magnetization time signal for linear flow is shown below.

![Linear Flow-Induced Adiabatic Inversion - PCASL Bloch Sim Magnetization vs Time](./poster_figures/Linear%20Flow-Induced%20Adiabatic%20Inversion%20-%20PCASL%20Bloch%20Sim%20Magnetization%20vs%20Time.png "Linear Flow-Induced Adiabatic Inversion - PCASL Bloch Sim Magnetization vs Time")

### Quadratic Flow

Quadratic or constantly accelerating blood flow is modelled in `Bloch PCASL Quadratic Trajectory With Aliases.ipynb`. Note the increase in speed causes the bolus to cross several aliased labelling planes, as shown below.

![Constantly Accelerating Flow and Aliased Labelling Planes](./poster_figures/Constantly%20Accelerating%20Flow%20and%20Aliased%20Labelling%20Planes.png "Constantly Accelerating Flow and Aliased Labelling Planes")

### Carotid Flow

Systolic-diastolic blood flow velocity in the carotid artery is modelled in `Bloch PCASL Systolic-diastolic Carotid Trajectory.ipynb`, and shown below.

![Carotid Flow with a Slow Bolus Crossing the Labelling Plane](./poster_figures/Carotid%20Flow%20with%20a%20Slow%20Bolus%20Crossing%20the%20Labelling%20Plane.png "Carotid Flow with a Slow Bolus Crossing the Labelling Plane")

### Aortic Flow

Systolic-diastolic blood flow velocity in the aorta is modelled in `Bloch PCASL Systolic-diastolic Aortic Trajectory.ipynb` and `Bloch PCASL Systolic-diastolic Aortic Trajectory - Postion Offset.ipynb`, and shown below for two boluses, moving fast and slow across the labelling plane.

![Aortic Flow with a Fast Bolus Crossing the Labelling Plane](./poster_figures/Aortic%20Flow%20with%20a%20Fast%20Bolus%20Crossing%20the%20Labelling%20Plane.png "Aortic Flow with a Fast Bolus Crossing the Labelling Plane")

![Aortic Flow with a Slow Bolus Crossing the Labelling Plane](./poster_figures/Aortic%20Flow%20with%20a%20Slow%20Bolus%20Crossing%20the%20Labelling%20Plane.png "Aortic Flow with a Slow Bolus Crossing the Labelling Plane")

Built atop a previous project from 2022:

# EECE 597 Engineering Demo

## Comparing kinetic models and Bloch simulations for quantitative perfusion imaging with Arterial Spin Labelling

### University of British Columbia - Electrical and Computer Engineering

This repo houses Jupyter notebooks comprising the EECE 597 demo. `Bloch ASL PCASL-GRASE.ipynb` encapsulates the majority of the Bloch simulations prototyped in other notebooks. `Kinetic ASL Model.ipynb` contains a CASL model shown in an interactive plot for adjusting model parameters in real-time. All figures produced for the EECE 597 report can be found in the top-level notebooks.

