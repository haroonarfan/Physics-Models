# Physics Projects

Below are some projects I did to consolidate my basics.

## Kinematics
  1. ### Problem Statement
     Modeling the behaviour of particles in different situations.

  2. ### Motivation
     To consolidate my understanding of the topics covered in class and improve my visualisation and computing skills.

  3. ### Objectives
     1. Accurately model parabolic and other situations involving forces and accelerations.
     2. Use acceleration to find the unknown forces
     3. Consider drag and forces due to wind, engine, mistic etc

  4. ### Implementation
     Everything exists inside the `Space` class. Everything is a `Particle`. If a force is added or updated then acceleration is updated automatically. The `predict_pos_in_time` and `predict_velocity_in_time` update the  position and velocity of the particle in time.
        > ### Reserved positions in `Particle.forces`
        > | Forces | Row |
        > |:------:|:---:|
        > | Weight |  0  |
        > |  Drag  |  1  |
        > These **labels** are saved as *strings* in `Particle.forces_labels`

#### Status: `In Progress`

## Results

   1. ### Model Performance
      - Kinematics
          * Here, I will present the performance of the model on the test set.
      - Springs
          * Here, I will present the performance of the model on the test set.

   2. ### Model Interpretation
         This section will explain how I interpreted the model.

   3. ### Model Visualization
       Here, I will present the visualization of the model.

#### Status: `Future`
