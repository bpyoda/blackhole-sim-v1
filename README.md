BLACK HOLE MERGER SIMULATION ("Ultimate Spectral Chaos": N-Body Problem, Black Hole Merger)

This black hole simulation is a real-time 3D gravitational field simulation written in Python using Taichi. The project focuses on visualizing orbital mechanics by combining an N-body gravity simulation with a fixed pool of massless tracer particles. Every trajectory emerges from the gravitational interactions between the massive bodies.

> Light mode and.. heavy mode 

The simulation presents a cinematic view of space with an orbiting camera, a subtle starfield, and two rendering modes.
Light Mode depicts massive black hole bodies in a light, non-consuming render so you can marvel at the physics at work. The focus is on the particles and their attraction to the black holes, also showing the gravitational lensing at work. You can make it stronger/weaker in the physics control panel located on the right.
While Heavy Mode renders the black holes with glowing accretion disks, gentle gravitational lensing, and event horizons. Together with additive glow, particle trails, and a restrained colour palette, these visualizations create depth without relying on computationally expensive shading techniques.

> Computing the real (astro)physics

The underlying simulation is driven entirely by real Newtonian physics. Black holes interact through Newton's Law of Universal Gravitation, with their motion integrated using the detailed algorithm to maintain stable long-term orbits. The implementation also demonstrates concepts such as gravitational potential, conservation of momentum and energy, Keplerian orbital motion, escape velocity, Hill spheres, Lagrange regions, and the restricted three-body problem.

> Particle chaos (what do the colors mean)

Gravitational force itself is visualized using a fixed pool of 1,000 massless tracer particles. These particles are influenced by every black hole but do not affect the simulation, allowing them to reveal orbital paths, gravitational streams, resonances, chaotic regions, and capture trajectories.

Each particle gradually transitions in a rainbow progression as it ages before fading away, creating a constantly evolving representation of the invisible structure of the gravitational field. Red particles are high energy and as they transform into blue particles, they lose energy and slowly fade away, simulating a simplified version of the life cycle of main sequence stars. The red particles are newly born stars and blue particles are fading white dwarfs. 

The black hole grows slightly when consuming the relatively tiny stars to the black hole's size and density.

> How the simulation works

Launch the simulation to begin with a stable binary black hole system. 

Left-click anywhere to create a new black hole. This black hole will interact with the existing stable binary system that has dual black holes and show the complex concept behind the 3 body problem.

Right-click and drag to orbit the camera, and use the mouse wheel to zoom in and out.

The sidebar provides controls for switching between Light and Dark rendering modes, adjusting visual quality, and changing the simulation speed. 
To change the physics in this simulation, press the button in the bottom left and open the physics control panel. You can adjust variables using the sliders.

THANKS FOR READING!! HOPE YOU ENJOY THE SIMULATION
