# SDO image reconstruction

This *Markdown* file is here to explain the method used to reconstruct an SDO image given the 3D reconstruction of a solar protuberance. 

To do so, we can define the problem as follows:\
 - 1: We need to define the positions of the 3D voxels in an orthonormal frame centred on SDO's position. Furthermore, we can define the new z-axis ($\mathbf{\vec{z}'}$) as going from 
SDO's position to the center of the Sun (being the origin of the initial frame). Lastly, keeping in mind the right hand rule and how SDO's image's are positioned, we need
to have the new x-axis ($\mathbf{\vec{x'}}$) parallel and in the same direction than the initial z-axis (i.e. in the direction of the Sun's north pole). Hence, the new y-axis ($\mathbf{\vec{y}'}$) need to be orthogonal to the initial z-axis. \
 - 2: We then need to change the reference frame centred on SDO to a spherical one to be able to get the voxel positions as a function of angles ($\mathbf{\theta'}$ and $\mathbf{\phi'}$). From there, knowing the number of pixels in SDO's CDD (i.e. 4096*4096) and the angular resolution of each pixel, we can define the borders of our hypothetical image
centred on the Solar disk. Finally, binning the $\theta'$ and $\phi'$ values given SDO's angular resolution, we recreate SDO's acquisition if it had been looking at the solar protuberance. 
 - 3: The last step is to then warp this image to a polar one to be able to incorporate it in our final re-projection plot.


## Equations:

1: We can define the first part of the problem with the following equations:

Orthonormal basis:
$$
(1.a) \space \space \vec{x}' \cdot \vec{z}' = 0 \\
(1.b) \space \space \vec{y}' \cdot \vec{z}' = 0 \\
(1.c) \space \space \vec{x}' \cdot \vec{y}' = 0
$$
$\mathbf{\vec{y}'}$ being orthogonal to the initial z-axis:
$$
(1.d) \space \space \vec{y}' \cdot \vec{z} = 0
$$
The new frame following the *"right hand"* convention:
$$
(1.e) \space \space \vec{x}' \times \vec{y}' = \vec{z}'
$$
$\mathbf{\vec{x}'}$ being in the same direction than the initial z-axis:
$$
(1.f) \space \space \vec{x}' \cdot \vec{z} > 0
$$

By defining the new vectors to be: 
$$
\vec{z}' = a \vec{x} + b \vec{y} + c \vec{z} \\ 
\vec{x}' = a_{x'} \vec{x} + b_{x'} \vec{y} + c_{x'} \vec{z} \\
\vec{y}' = a_{y'} \vec{x} + b_{y'} \vec{y} + c_{y'} \vec{z}
$$
we can start finding the solution for this system knowing **a**, **b** and **c** (as $\mathbf{\vec{z}'}$ is just $\mathbf{-\overrightarrow{SDOpos}}$).

From **(1.d)**, we get
$$
(1.g) \space \space c_{y'} = 0
$$
Using **(1.g)** and **(1.b)**, we have
$$
a_{y'} a + b_{y'} b = 0 \\ 
(1.h) \space \space a_{y'} = - \frac{b}{a} b_{y'}
$$
From **(1.h)**, **(1.c)** and **(1.g)**, we can deduce that
$$
a_{x'} a_{y'} - \frac{a}{b} a_{y'} b_{x'} = 0 \\
(1.i) \space \space a_{x'} = \frac{a}{b} b_{x'}
$$
From **(1.a)** and **(1.i)**, we can complete $\mathbf{\vec{x}'}$
$$
a_{x'} a + \frac{b^{2}}{a} a_{x'} + c_{x'} c = 0 \\
c_{x'} = - \frac{a^{2} + b^{2}}{a c} a_{x'}
$$
From **(1.e)**, we can find a relation between $\mathbf{\vec{x}'}$ and $\mathbf{\vec{y}'}$
$$
\vec{y}' \times \vec{z}' = \vec{x}' \\
a_{x'} = b_{y'} c - 0 * b \\
b_{y'} = \frac{1}{c} a_{x'} 
$$
Putting all the final equations together, we get that:
$$
\vec{z}' = a \vec{x} + b \vec{y} + c \vec{z} \\
\vec{x}' = a_{x'} \vec{x} + \frac{b}{a} a_{x'} \vec{y} - \frac{a^{2} + b^{2}}{a c} a_{x'} \vec{z} \\
(1.j) \space \space \space \space \vec{y}' = - \frac{b}{ac}a_{x'} \vec{x} + \frac{1}{c} a_{x'} \vec{y}
$$
Now, to define all the vector components just with **a**, **b** and **c**, we need to take into account the equations giving the direction of the new frame. \
In that regard, by combining the appropriate equation of **(1.j)** and **(1.f)**
$$
\frac{a^{2} + b^{2}}{a}a_{x'} > 0
$$
We conclude that there are two different systems:
 - if **a > 0**: $\space \space \mathbf{a_{x'} > 0}$.
 - if **a < 0**: $\space \space \mathbf{a_{x'} < 0}$.

For the normalisation we can admit that $\mathbf{a_x' = \pm 1}$
$$
\|\vec{x}'\|^{2} = 1 + \frac{b^2}{a^2} + (\frac{a^2 + b^2}{a c})^2 \\
\|\vec{y}'\|^{2} = (\frac{b}{ac})^2 + \frac{1}{c^2} \\
N_{x'} = \frac{1}{\sqrt{1 + \frac{b^2}{a^2} + (\frac{a^2 + b^2}{a c})^2}} \\
N_{y'} = \frac{ac}{\sqrt{a^2 + b^2}} \\
N_{z'} = \frac{1}{\sqrt{a^2 + b^2 + c^2}}
$$

 - for **a > 0**, we can admit that $\mathbf{a_x' = 1}$, hence
$$
\hat{\vec{x}}' = N_{x'} (\hat{\vec{x}} + \frac{b}{a} \hat{\vec{y}} - \frac{a^2 + b^2}{a c} \hat{\vec{z}}) \\

\hat{\vec{y}}' = N_{y'} (\frac{-b}{ac} \vec{x} + \frac{1}{c} \vec{y})\\

\hat{\vec{z}}' = N_{z'} (a \hat{\vec{x}} + b \hat{\vec{y}} + c \hat{\vec{z}})
$$
 - for **a < 0**, we can easily see that
$$
\hat{\vec{x}}' = -N_{x'} (\hat{\vec{x}} + \frac{b}{a} \hat{\vec{y}} - \frac{a^2 + b^2}{a c} \hat{\vec{z}}) \\

\hat{\vec{y}}' = -N_{y'} (\frac{-b}{ac} \vec{x} + \frac{1}{c} \vec{y})\\

\hat{\vec{z}}' = N_{z'} (a \hat{\vec{x}} + b \hat{\vec{y}} + c \hat{\vec{z}})
$$

Now, to get the new coordinates of a voxel ($\mathbf{x'_{voxel}}$, $\mathbf{y'_{voxel}}$, $\mathbf{z'_{voxel}}$), we know that
$$
\overrightarrow{voxel}' = \vec{x}' + \vec{y}' + \vec{z}' + \overrightarrow{voxel}
$$
and the coordinates in the new reference frame for the voxel are
$$
x'_{voxel} = \overrightarrow{voxel}' \cdot \hat{\vec{x}}' \\
y'_{voxel} = \overrightarrow{voxel}' \cdot \hat{\vec{y}}' \\
z'_{voxel} = \overrightarrow{voxel}' \cdot \hat{\vec{z}}'
$$
Therefore
$$
x_{voxel}' = \vec{x}' \cdot \hat{\vec{x}}' + x_{voxel} \hat{\vec{x}} \cdot \hat{\vec{x}}' + y_{voxel} \hat{\vec{y}} \cdot \hat{\vec{x}}' + z_{voxel} \hat{\vec{z}} \cdot \hat{\vec{x}}' \\
x_{voxel}' = \|\vec{x}'\| + sign(a) N_{x'} (x_{voxel} + y_{voxel} \frac{b}{a} - z_{voxel} \frac{a^2 + b^2}{a c}) \\
x_{voxel}' = \frac{1}{N_{x'}} + sign(a) N_{x'} (x_{voxel} + y_{voxel} \frac{b}{a} - z_{voxel} \frac{a^2 + b^2}{a c})
$$
Using the same reasoning, we get
$$
y_{voxel}' = \frac{1}{N_{y'}} + sign(a) N_{y'} (x_{voxel} \frac{-b}{ac} + y_{voxel} \frac{1}{c}) \\
z_{voxel}' = \frac{1}{N_{z'}} + sign(a) N_{z'} (x_{voxel} a + y_{voxel} b + z_{voxel} c)
$$

2: Having redefined the voxel positions in our new reference frame centred on SDO, we can now start the corresponding image reconstruction. \
To do so we first need to change our frame to the corresponding spherical coordinate system. \
We know that
$$
\hat{\vec{r}}' = sin(\theta) cos(\phi) \hat{\vec{x}}' + sin(\theta) sin(\phi) \hat{\vec{y}}' + cos(\theta) \hat{\vec{z}}' \\
\hat{\vec{\theta}}' = cos(\theta) cos(\phi) \hat{\vec{x}}' + cos(\theta) sin(\phi) \hat{\vec{y}}' - sin(\theta) \hat{\vec{z}}' \\
\hat{\vec{\phi}}' = - sin(\phi) \hat{\vec{x}}' + cos(\phi) \hat{\vec{y}}'
$$
Where the radius $\mathbf{r}$, the polar angle $\mathbf{\theta} \in [0, \pi]$ and the azimuth $\mathbf{\phi} \in [0, 2\pi[$ defined as 
$$
r = \sqrt{\text{x'}^2 + \text{y'}^2 + \text{z'}^2} \\
\theta = arcos(\frac{z'}{\sqrt{\text{x'}^2 + \text{y'}^2 + \text{z'}^2}}) \\
\phi = sgn(y') arcos(\frac{x'}{\sqrt{\text{x'}^2 + \text{y'}^2}})
$$
Omitting the object's distance to SDO (i.e. **r**), we can now define the solar protuberance as a function of $\mathbf{\theta}$ anf $\mathbf{\phi}$.

3: Having already defined that $\mathbf{\vec{y}'}$ is perpendicular to $\mathbf{\vec{z}}$ and that $\mathbf{\vec{x}'}$ is positive when $\mathbf{\vec{z}}$ positive, it is clear that our $\mathbf{\theta}$ and $\mathbf{\phi}$ respectively represent $\mathbf{\rho_{image}}$ $\mathbf{\theta_{image}}$ in the polar coordinate system centred on the center of the Sun's disk, as seen by SDO (as $\mathbf{\theta}$ is the angle going from the disk center to the object and $\mathbf{\phi}$ the angle between $\mathbf{\vec{z}}$ and the object). \
To reconstruct the SDO image, we then just need to take into account SDO's polar angle and radial distance resolution when an abject is at $\mathbf{\|\overrightarrow{SDOpos}\|}$ distance (for the radial distance resolution as SDO's polar angle resolution is just SDO's angle resolution).