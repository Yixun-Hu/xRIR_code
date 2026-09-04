# arXiv:2504.10746v2 [cs.CV] 4 Jun 2025

## Hearing Anywhere in Any Environment

### Xiulong Liu^1 * Anurag Kumar^2 Paul Calamia^2 Sebastia V. Amengual`^2 Calvin Murdock^2

### Ishwarya Ananthabhotla^2 Philip Robinson^2 Eli Shlizerman^1 Vamsi Krishna Ithapu^2 Ruohan Gao^3

(^1) University of Washington (^2) Meta (^3) University of Maryland, College Park

### Abstract

```
In mixed reality applications, a realistic acoustic experi-
ence in spatial environments is as crucial as the visual ex-
perience for achieving true immersion. Despite recent ad-
vances in neural approaches for Room Impulse Response
(RIR) estimation, most existing methods are limited to the
single environment on which they are trained, lacking the
ability to generalize to new rooms with different geometries
and surface materials. We aim to develop a unified model
capable of reconstructing the spatial acoustic experience of
any environment with minimum additional measurements.
To this end, we presentXRIR, a framework for cross-room
RIR prediction. The core of our generalizable approach lies
in combining a geometric feature extractor, which captures
spatial context from panorama depth images, with a RIR
encoder that extracts detailed acoustic features from only a
few reference RIR samples. To evaluate our method, we in-
troduceACOUSTICROOMS, a new dataset featuring high-
fidelity simulation of over 300,000 RIRs from 260 rooms.
Experiments show that our method strongly outperforms a
series of baselines. Furthermore, we successfully perform
sim-to-real transfer by evaluating our model on four real-
world environments, demonstrating the generalizability of
our approach and the realism of our dataset.
```
### 1. Introduction

```
Each environment echoes its own story, creating a distinct
auditory experience. Imagine walking through a museum
where each room’s unique acoustic character brings ex-
hibits to life, immersing you in stories through tailored
soundscapes. To recreate such realistic auditory experi-
ences across different spaces, models must seamlessly and
easily adapt from one environment to another. Thiscross-
roomadaptability is essential for applications like virtual re-
ality and immersive media, where authentic acoustics trans-
form and enhance how we experience sound anywhere in
any environment of interest.
*Work done during the internship at Meta.
```
```
Figure 1. OurXRIR framework can predict accurate room impulse
responses (RIR) of any new environment, by integrating the geo-
metric prior learned from a large simulated RIR dataset of diverse
training environments and the nuanced acoustic profile extracted
from a few reference RIR measurements in the new environment.
```
```
To model how sound interacts with an environment, a
room impulse responses (RIR) is often measured to capture
how a perfect impulse emitted from the source location re-
flects, is absorbed, diffuses, and gets received at the micro-
phone location, all according to the room’s unique geometry
and surface materials. Traditionally, to fully capture the im-
mersive acoustic field in an environment, hundreds of RIR
measurements are gathered by positioning speakers and mi-
crophones densely throughout the space [32, 38, 43, 51].
However, this process is labor-intensive and costly, espe-
cially when scaling across diverse real-world environments.
Recent deep learning approaches [10, 29, 32, 51] lever-
age implicit neural networks to “compress” these dense RIR
measurements into a single model that can be queried for
RIR at any source-listener pair in a specific room. How-
ever, training these models still requires a large amount of
densely sampled RIRs for each room, as they are designed
to overfit to asingle room’s specific geometry and material
properties. When applied to a new environment, these mod-
els must be re-trained with a similarly dense dataset, limit-
ing their practicality for scalable cross-room applications.
Our goal is to develop a model that can handle variations
in room geometry and surface material properties, which
are key factors in shaping each room’s unique acoustic pro-
file. Achieving this goal presents several challenges. First,
```

the model should utilize an easily obtainable and standard
visual representation of the environment to extract geomet-
ric properties from any room. Second, given the diverse
and nuanced acoustic properties of different surface materi-
als, we need a way to quickly capture essential cues about a
room’s detailed acoustic characteristics (e.g., energy decay
and reverberation patterns). Third, to ensure generalization,
we need a large-scale and high-fidelity RIR dataset that en-
compasses a wide range of room environments with varied
acoustic materials and geometries, enabling the pre-training
of a cross-room feature extractor.
To address these challenges, we introduce XRIR, a
generalizable model for cross-room RIR prediction, along
with ACOUSTICROOMS, a large dataset comprising over
300,000 realistic RIRs simulated from 260 rooms, specif-
ically curated for this task. Our model features three key
components: i) a Geometric Feature Extractor, which uti-
lizes a vision transformer to process panorama depth images
from the receiver’s perspective, capturing the spatial rela-
tionships between the source and receiver positions within
the room; ii) a Reference RIR Encoder, which extracts
spatio-temporal features from a few reference RIRs, cap-
turing the unique energy decay and reverberation character-
istics associated with room materials; and iii) a Fusion and
Weighting Module, which predicts the target RIR through
a weighted combination of the reference RIRs. By inte-
grating complementary geometric and acoustic features, our
model effectively approximates both structural and material
properties of any room, enabling precise RIR predictions
not only at new locations in the training environments, but
also in any new environment of interest.
We evaluate our model’s performance in both seen and
unseen environments from ACOUSTICROOMS, demonstrat-
ing its capability to predict RIRs at new locations within
known rooms and also effectively generalize to entirely new
environments. Our method consistently achieves state-of-
the-art results across these scenarios, outperforming sev-
eral strong baselines and prior methods. In addition, to as-
sess the model’s real-world applicability, we successfully
perform sim-to-real transfer by deploying our model on
a dataset comprising four real-world environments. This
demonstrates the effectiveness of our framework and also
the realism of RIR simulations in our dataset.
In sum, our main contributions are as follows:

- We proposeXRIR, a cross-room generalizable framework
    that predicts accurate RIRs for any seen and unseen envi-
    ronment, strongly outperforming prior methods.
- We introduce ACOUSTICROOMS, a large-scale dataset tai-
    lored for this task, comprising over 300,000 high-fidelity
    RIRs simulated from 260 diverse rooms.
- Apart from superior performance on simulated rooms, we
    also successfully deploy our model in four real-world en-
    vironments, showcasing effective sim-to-real transfer.

### 2. Related Work

```
Learning-Based RIR Prediction. Early machine learn-
ing methods for room impulse response (RIR) prediction,
such as Image2Reverb [48] and Fast-RIR [40], utilize a
generative approach conditioned on semantic information
like RGB images of the environment, source and listener
locations, and T60 values. While these approaches pro-
duce plausible RIRs aligned with scene semantics and ba-
sic acoustic constraints, they struggle to reproduce accurate
RIRs at arbitrary locations within the target scene. Recent
advances on implicit neural representations [34, 49] have
inspired a series of works that approximate a function map-
ping spatial coordinates of source and listener locations to
RIRs [32, 39, 43, 51]. Some methods [1, 2, 8, 10, 24, 28, 41]
also condition on room geometry and material properties
of the visual environment. By explicitly modeling the 3D
scene, these methods can render precise RIRs at novel lo-
cations within the same environment they are trained on.
However, they lack the ability to generalize to new environ-
ments, which is the focus of our work.
Closest to our work are Diff-RIR [56] and Few-Shot
RIR [33], both of which also address cross-room RIR gen-
eralization. Diff-RIR learns material coefficients through a
differentiable rendering framework based on planar room
geometry and a few RIR measurements, enabling RIR ren-
dering at any location using the image source method. How-
ever, it requires training a separate model for each room,
making it computationally intensive and less scalable for
large or complex environments. Few-Shot RIR leverages a
limited number of RIR measurements and RGB-D obser-
vations to predict RIRs at new locations by integrating fea-
tures from pose, RGB-D data, and binaural echoes. How-
ever, it does not effectively utilize geometric information as
our method and is only trained and tested in simulation. We
compare our approach with both methods in experiments.
Room Acoustics Simulation. Room acoustic simulation
is crucial for applications in AR/VR [25], architectural de-
sign [36], and far-field speech recognition [22, 37], serv-
ing as a bridge between simulated and real-world acoustics.
Approaches can be generally categorized as wave-based,
geometric-based, or hybrid. Wave-based methods [13] ac-
curately capture low-frequency phenomena but are com-
putationally intensive. Geometric-based methods, such
as [3, 6, 45, 46], efficiently trace sound rays but lack ac-
curacy at low frequencies. Hybrid methods [53] combine
both approaches to balance accuracy and efficiency. We
utilize the hybrid approach [42] from the Treble simulation
platform to simulate high-fidelity RIRs for our cross-room
RIR prediction task, optimizing both accuracy and compu-
tational demands.
Audio-Visual Learning of Room Acoustics. Both vision
and audio provide significant spatial information that re-
```

```
veals room properties. Prior methods have combined both
modalities for a series of audio-visual learning tasks re-
lated to room acoustics, including audio spatialization us-
ing visual spatial cues from the environment [16, 19, 20,
26, 35], audio-visual navigation in environments with vary-
ing room acoustic properties [3, 4, 15, 18, 30], learning
image features, scene structures, or human locations from
echoes [12, 17], ambient sound [9], or music [55] in the
room, and using RGB images or videos of a target envi-
ronment to guide sound transfer that aligns with the space’s
acoustics [5, 7, 11, 27, 50]. Our work also integrates both
visual and audio information, but we address a different
challenging task that aims to infer accurate room acoustics
in any new environment.
```
### 3. Our Approach

#### 3.1. Problem Formulation

We tackle thecross-roomroom impulse response (RIR) pre-
diction task, which aims to predict single-channel (omnidi-
rectional) RIRs for any source-receiver pair across diverse
room environments, including thoseunseenduring training.
We aim to develop a generalizable model that can accurately
predict RIRs in any environment without labor-intensive
data collections or training a separate model for each room.
Our model (detailed in Sec. 3.2) achieves this by utilizing
only minimal additional measurements from the new room,
such as only a few panorama depth images and reference
RIR measurements, to quickly adapt to new acoustic envi-
ronments with minimal effort, thereby facilitating general-
ization to previously unseen environments. Next, we for-
mally define the cross-room RIR prediction task by outlin-
ing its data, inputs, and the modeling objective.
Data.LetR={R 1 ,R 2 ,...,RM}represent a dataset of
Mrooms, split into a training set,Rtrain⊂R, and a test set,
Rtest=R\Rtrain. Each roomRmincludes a set of receiver
locations, denotedLm={Pr(m,1),Pr(m,2),...,Pr(m,Nm)},
whereNmis the number of receivers in the room. For each
receiverPr(m,i)in roomRm, RIRs are measured at various
source locationsPs(m,i,j), resulting in measurementsAm,i,j

of the source-receiver pairP
(m,i,j)
s andP

(m,i)
r.
Inputs.To capture the necessary observation conditions
for predicting a target RIRAt, we define an observation
tupleO= (Ps,Pr,Gr), wherePsis the target source loca-
tion,Pris the receiver location, andGrrepresents the local
geometry near the receiver locationPr, e.g., room bound-
ary points or depth maps around the receiver. Additionally,
we introduce a set ofKreference RIRs measured at the tar-
get receiver locationPrfrom various reference source lo-

cationsPref,s={Pref,s(1),Pref,s(2),...,Pref,s(K)}. These references,

denoted asAref={A(1)ref,A(2)ref,...,A(refK)}, are crucial for
capturing essential acoustic characteristics that encode nu-
anced information about room materials. Note that in the

```
above formulation, while we fix the receiver location and
set reference RIRs at different source locations, exchang-
ing the receiver and source in the input yieldsan equivalent
alternative formulation.
Modeling Objective.The objective of cross-room RIR
prediction is to train a modelF that predicts the target
RIRAˆtusing the observation tupleOalong with the refer-
ence RIRsArefand their respective source locationsPref,s:
Aˆt=F(O,Aref,Pref,s). In this formulation,Oprovides
the geometric and positional context, whileArefandPref,s
give sparse acoustic observations that help bridge the lack
of explicit material properties by capturing key room acous-
tics characteristics.
Unlike thesingle-roomRIR prediction task [32, 51],
which assumes consistent geometry and material properties
and fits a separate model for each scene, our cross-room
formulation aims to train a single model that generalizes
across multiple scenes with diverse room geometries and
materials, while with the extra condition of only a few RIR
measurements. Models that are designed for single-room
RIR prediction task [32, 51] must be re-trained with dense
data when applied to a new environment. While our method
uses one unified model to predict accurate RIRs across dif-
ferent rooms, seen or unseen. Our formulation can also be
easily adapted to the single-room RIR prediction setting by
fitting dense measurements in the room as in prior work.
Please see supp. for results on single-room experiments.
```
#### 3.2. TheXRIRModel

```
To solve the cross-room RIR prediction task, we propose a
new architecture,XRIR, which processes not only geome-
try and positional features of source and receiver, but also
leverages the reference RIRs to accurately predict the target
RIR. As illustrated in Fig. 2,XRIR consists of three main
components: i) aGeometric Feature Extractor(Sec. 3.2.1),
which encodes the spatial relationships among the source,
receiver, and room surface geometry, capturing important
geometric features that shape acoustic behavior; ii) aRef-
erence RIR Encoder(Sec. 3.2.2), which processes the spa-
tiotemporal characteristics of the reference RIRs to extract
features that represent their acoustic properties within the
room.; and iii) aFusion and Weighting Module(Sec. 3.2.3),
which integrates the spatial features from the Geometric
Feature Extractor with the reference RIR features from the
Reference RIR Encoder, generating a set of weights to com-
bine reference RIRs as the predicted target RIR.
3.2.1. Geometric Feature Extractor
The Geometric Feature Extractor module captures spatial
relationships among the source, receiver, and room ge-
ometries, which is important for accurate acoustic model-
ing. It consists of two geometric sub-modules: theDirect
Path Module, and theReflection Module. These two mod-
ules emulate the process of sound propagation. The Direct
```

```
Figure 2.System Overview of OurXRIR Model for Cross-Room RIR Prediction. The model architecture consists of three main
components: i) aGeometric Feature Extractor, which captures spatial relationships among the source, receiver, and room geometry; ii)
aReference RIR Encoder, which extracts spatiotemporal features from reference RIRs; and iii) aFusion and Weighting Module, which
integrates these spatial and acoustic features to predict the target RIR.
```
```
Figure 3.Illustration of the Geometric Feature Extractor.Rec:
Receiver, Tgt Src: Target Source, Ref Src: Reference Source.
```
```
Path Module extracts the feature of the direct path between
source and receiver, while the Reflection Module models
the sound propagation path through the reflections from the
room boundaries. A detailed overview of the Geometric
Feature Extractor is illustrated in Fig. 3.
Direct Path Module.To capture the direct path between
each source and the receiver, we concatenate their 3D coor-
dinates. For the target source, we definePdir= (Ps,Pr),
wherePsandPrare the coordinates of the source and the
receiver, respectively. For each reference sourcePref,dir(k) ,
we defineP
(k)
ref,dir = (P
```
```
(k)
ref,s,Pr). Pdirand eachP
```
(k)
ref,diren-
code locations of every source-receiver pair, thereby encod-
ing the direct path information. To extract their features,
we apply sinusoidal positional encoding [54] followed by
a multi-layer perceptron (MLP) to project them into high-
dimensional vectors, resulting ingdir∈R^1 ×Cdfor the target

source-receiver pair andg(ref,dirk) ∈R^1 ×Cdfor each reference
source-receiver pair.
Reflection Module. Inspired by INRAS [51], we also
model the reflection paths between source and receiver via
the room boundary. But differently, instead of using a
fixed set of bounce points per room, we propose to use a
panorama depth map at the receiver’s location as a proxy

```
for local room geometry to unify the representation across
different rooms.
Given the panorama depth mapIdp ∈RH×W×^3 cen-
tered at receiver’s viewpoint, we first projectIdpto a 3D
coordinate mapIcoord=Frect(Idp)via an equi-rectangular
projection transformationFrect. Each pixel inIcoordrepre-
sents the 3D coordinate of a visible boundary point in the
room from the receiver’s view. Each source in the room has
chance to reflect through these points until finally reaching
the receiver. To model such interactions, we create a set
of reflections-based maps by subtracting the 3D coordinate
map from the source and receiver positions.
To perform the subtraction, it is necessary to unify the
coordinates between the 3D coordinate map and the source
/ receiver positions. We achieve this by projecting the world
coordinates of the sources and the receiver into camera co-
ordinates at the receiver’s position, resulting in the same
coordinate system as the 3D coordinate map. We obtain the
target source position asPrel,s=R(Ps−Pr)and each ref-
erence source location asPrel, ref(k) =R(Pref(k,s)−Pr), whereR
is the world-to-camera transformation matrix. Then we cre-
ate reflection-based mapsIrf,sfor the target source,Irf, ref(k)
for reference sources, as well asIr,rffor the receiver by
performing subtractions: Is,rf =Prel,s−Icoord,I
(k)
ref,rf =
Prel, ref(k) −IcoordandIr,rf=Icoord− 0 , where 0 means the
origin, where the receiver is located.
These reflection-based maps encode the dense interac-
tion between room geometry and the sources / receiver. To
further extract features, we utilize a vision transformer mod-
ule [14]Fvtthat partitions each reflection-based map into
patches, aggregates local features, and builds spatial depen-
dencies among patches. This results in compact patch-level
geometry representations:g′r,rf=Fvt(Ir,rf),g′s,rf=Fvt(Is,rf)
and{g′ref,rf(k)=Fvt(I(ref,rfk))}Kk=1, where each feature map has
```

```
dimensionalityNp×Cp. Finally, we apply a MLP layer to
project the patch dimensionNpto 1, resulting ingr,rf,gs,rf,
```
and{g(ref,rfk)}Kk=1, respectively.

3.2.2. Reference RIR Encoder

To capture acoustic features related to energy decay and re-
verberation patterns within the room, we leverage reference
RIRs as proxies for the acoustic characteristics at various
source locations relative to the receiver. To encode these
acoustic features, we first compute the log-magnitude spec-
trogram of each reference RIR using the Short-Time Fourier
Transform (STFT):Sref,k = log(∥STFT(Aref,k)∥), where
Sref,k∈RF×T. To extract robust acoustic features, we im-
plement the Reference RIR Encoder using ResNet-18, and
use the mean pooled featuresfa(k)∈Rdfrom the last layer
to encode each reference RIR.

3.2.3. Fusion and Weighting Module

The Fusion and Weighting Module integrates the outputs
from the Geometric Feature Extractor and the Reference
RIR Encoder to generate the target RIR prediction. This
module combines geometric and acoustic features for ref-
erence sources as well as the geometric features of target
source, finally computing the weights that are applied to
reference RIRs.
Fusion of Geometric and Acoustics Features. For
each reference source, we combine the geometric feature
gref,dir(k) , g(ref,rfk), gr,rfand the acoustic featurefa(k)by con-
catenating them along the feature dimension, resulting in:

h(refk)=Concat(gref,dir(k) ,gref,rf(k),gr,rf,fa(k)).
Similarly, for the target source, we combine the geomet-
ric featuregdir,gs,rfandgr,rfvia concatenation, yielding:
h′t=Concat(gdir,gs,rf,gr,rf). We then project the fused fea-
tureh′ttohtthrough a MLP to make the feature dimension

the same ash
(k)
ref.
To align the target and reference features, we compute
the attention between the target fused vectorhtand each
reference fused vectorh(refk). Specifically, given the refer-

ence fused featuresHref={h
(k)
ref}

K
k=1and the target fused
vectorht, the attention outputZis computed as:

```
Z=softmax
```
##### 

```
Href·hTt
√
C
```
##### 

```
⊙Href,
```
where·and⊙denote matrix multiplication and element-
wise multiplication with broadcasting respectively, and
Href ∈RK×C,ht∈R^1 ×C,Z∈RK×C. These atten-
tion outputsZ={zk}Kk=1for each reference RIR is now
attended by the fused feature of the target RIR.
Time-Aligned Weighting Matrix. Given the attention
outputsZ∈RK×C, we next generate a time basis vec-
torTbbased on the temporal indices of the spectrogram
[0, 1 , 2 ,...,T]. Specifically, we computeT

′
busing sinu-
soidal positional encoding [54] and then apply a MLP layer

```
to projectT
```
```
′
btoC, resulting inTb∈R
```
```
T×C. We generate
the time-aligned weighting matrixW∈RK×Tby comput-
ing the outer product betweenZandTb:W=Z·TTb.
Each row ofWcorresponds to the weights applied to the
log-magnitude spectrogram of each reference RIR, adapt-
ing them to match the temporal structure of the target spec-
trogram. This weighting matrixWeffectively shapes each
reference spectrogram to align with the characteristics of
the target RIR.
Finally, we predict the target RIR’s log-magnitude spec-
trogramSpredvia the weighted sum of the log-magnitude
spectrograms of the reference RIRs:Spred=
```
##### PK

```
k=1Wk⊙
Sref,k.Wkis thek-th row of the weight matrixW, applied
to the corresponding log-magnitude spectrogramSref,k.
```
```
3.2.4. Training and Inference
During training, we use the magnitude STFTL 1 Loss to
compute the error between the magnitude spectrograms of
the predicted target RIR and the ground-truth RIR:LSTFT=
∥exp(Spred)−exp(Sgt)∥ 1. Additionally, following [33], we
incorporate an energy decay loss to optimize the decay pat-
terns of the predicted spectrogram. The energy decay loss
LEDis defined as: LED = ∥EDC(Spred)−EDC(Sgt)∥ 1 ,
where EDC(·)denotes the energy decay curve of RIR in
the frequency domain. The total loss becomesLtotal =
LSTFT+λLED, whereλis a weight to balance the contribu-
tion of the energy decay loss.
During inference, we randomly samples K RIRs
{Aref,k}kk==1K along with corresponding source locations
{Pref,k}kk==1Kfrom a test room as reference inputs. The
model predicts the magnitude spectrogram of a target RIR,
which is then converted back to a waveform via the Griffin-
Lim [21] algorithm.
```
### 4. The ACOUSTICROOMSDataset

```
To our best knowledge, there are two prior datasets with a
large number of rooms: SoundSpaces MP3D [3, 33] and
GWA [53]. SoundSpaces MP3D comprises only 83 rooms
with limited material variety (around 100 types), a fixed
one-to-one mapping between semantic labels and acous-
tic coefficients, and a constrained 2D configuration at fixed
heights. This setup restricts real-world applicability, as
actual rooms often contain diverse materials with varying
acoustic properties and require 3D spatial modeling to cap-
ture realistic sound propagation. For GWA, while it in-
cludes a large number of simulated RIRs from a wide va-
riety of synthetic rooms and explicitly models wave propa-
gation, the wave-based method it employs, PFFDTD [23], is
a lower-resolution approach. This method prioritizes com-
putational efficiency, which comes at the cost of reduced
simulation accuracy.
To address the above limitations, we introduce ACOUS-
TICROOMS, a new large-scale, high-quality dataset of sim-
```

ulated RIRs specifically designed for robust generalization
across diverse room geometries, sizes, and material prop-
erties. We use Treble Technology’s simulation platform^1 ,
where a more advanced wave-based solver, i.e., the Dis-
continuous Galerkin (DG) Method [42], is supported. Em-
ploying such techniques to simulate RIRs in our dataset is
crucial for achieving cross-room generalization and sim-to-
real transfer applications. ACOUSTICROOMSsimulates 260
rooms across 10 categories, featuring 300K simulated RIRs
from different source-receiver pairs and full 3D spatial con-
figurations. Each room includes a randomized material as-
signment from a library of 332 materials across 11 cate-
gories, ensuring diversity in acoustic properties even among
similar geometries. The combination of scale, material di-
versity, and simulation fidelity enables ACOUSTICROOMS
to accurately reflect the acoustics of real-world environ-
ments.

### 5. Experiments

#### 5.1. Implementation Details

In the ACOUSTICROOMSdataset, RIRs are sampled at
22,050 Hz with a maximum length of 9600 samples (0.
s). We compute the magnitude spectrogramSwith FFT
size 124, window size 62, and hop size 31, yielding a shape
of 63 × 310. Panorama depth maps of room geometry are
rendered at a resolution of 256 × 512 from the receiver’s
location, and source/receiver positions are recorded as 3D
coordinates(x,y,z). ForXRIR, we implement a Vision
Transformer blockFvtwith 6 multi-head attention layers
(8 heads, hidden size 512). Depth maps are divided into
16 × 32 patches, resulting in all reflection-based features
such asgr,rfandgs,rfof dimension 256 × 512. Direct path
features are calculated using sinusoidal encoding on each
3D coordinate with 20 frequency bins, and are then pro-
jected into 256-dimensional vectors via MLP. For loss cal-
culation, we setλ= 0. 01 to balance the STFT loss and the
energy decay loss.

#### 5.2. Baselines

We compare with a series of baselines as well as prior meth-
ods [33, 56]:

- Random Across Rooms: Randomly sample a RIR from
    the entire dataset as the prediction for the target RIR.
- Random Same Room: Randomly sample a RIR from the
    same room as as the prediction for the target RIR.
- Nearest Neighbor: Samplek-shot reference RIRs and
    select the RIR with the closest spatial distance to the tar-
    get source as the prediction.
- Linear Interpolation: Linearly interpolate betweenk-
    shot reference RIRs based on the distance between each
    reference and the target source location.

(^1) https://www.treble.tech/

- Few-Shot RIR [33]: Few-Shot RIR implements a trans-
    former architecture that fuses features from separate en-
    coders of multi-modal conditional inputs and then gener-
    ates the target RIR by decoding the transformer outputs
    via a UNet decoder. We adapt their model to our task
    by replacing the binaural echos with our single-channel
    reference RIRs (different source and receiver locations)
    and using panorama depth images as inputs to the image
    encoder instead of egocentric RGBD images.
- Diff-RIR [56]: We compare with Diff-RIR in evaluation
    on sim-to-real transfer. The framework utilizes the few-
    shot, i.e., 12 reference RIRs, to train a differentiable ren-
    dering pipeline to learn acoustics parameters of the room
    geometry. For fair comparison, we finetune ourXRIR
    model pre-trained on ACOUSTICROOMSon the same set
    of reference RIRs as Diff-RIR in each room, and then
    test on the same test split. Note that Diff-RIR requires
    training one model per each room and the training pro-
    cess becomes computationally infeasible for large space
    with complex room geometries. Therefore, we do not in-
    clude it in our comparison on ACOUSTICROOMS.
       In addition, for a more complete comparison with prior
methods on RIR prediction, we also adapt our method for
the single-room RIR prediction task and compare with prior
work [32, 51]. Please see Supp. for results.

#### 5.3. Metrics

```
We evaluate the energy pattern of the generated RIRs
against ground-truth RIRs using three key acoustic metrics,
which are strongly correlated with hearing perception and
commonly used in prior work on RIR prediction [32, 51]:
```
- Early Decay Time (EDT) Error:To evaluate early re-
    flection characteristics, we use the EDT error, which mea-
    sures the time taken for the initial 5 dB decay in the en-
    ergy curve.
- Clarity (C50):For comparing early-to-late energy ra-
    tios, we employ the clarity metricC 50 , which provides
    insights into the prominence of early reflections over later
    reverberations.
- T60Error:We evaluate the accuracy of reverberation
    time by comparing theT 60 value of the predicted RIR
    and the ground-truth RIR. We calculateT 60 usingT 20 ,
    based on a linear fit between− 5 dB and− 25 dB on the
    logarithmic energy decay curve obtained from Schroeder
    Backward Integration [47].

#### 5.4. Quantatitive Results onACOUSTICROOMS

```
We show cross-room RIR prediction results in both envi-
ronments seen during training as well as unseen new envi-
ronments. As shown in Table 1, ourXRIR model signifi-
cantly outperforms all baselines across all metrics (EDT er-
ror,C 50 error, andT 60 error). In the seen split, ourXRIR
model yields the lowest errors across metrics, particularly in
```

```
Method Seen Splits Unseen Splits
EDT error (s) C50 error (dB) T60 error (%) EDT error (s) C50 error (dB) T60 error (%)
Random Across Rooms 0.290 6.831 37.35 0.313 7.802 35.
Random Same Room 0.129 3.567 12.80 0.172 5.440 16.
Few-shot RIR [33] (K=1) 0.157 3.957 31.42 0.130 3.225 20.
Few-shot RIR [33] (K=4) 0.157 4.026 31.63 0.136 3.568 19.
Few-shot RIR [33] (K=8) 0.174 4.451 32.71 0.187 4.470 21.
Linear Interpolation (K=8) 0.094 2.421 9.76 0.121 3.090 13.
Nearest Neighbor (K=8) 0.064 1.717 8.94 0.090 2.667 11.
XRIR (K=1) 0.046 1.183 9.50 0.075 1.841 13.
XRIR (K=4) 0.040 1.005 8.15 0.068 1.335 13.
XRIR (K=8) 0.038 0.940 8.13 0.055 1.457 10.
```
Table 1.Cross-Room RIR Prediction Results for Both the Seen and Unseen Splits.We report EDT Error (EDT) in seconds, C50 Error
(C50) in dB, and T60 percentage error (T60), with lower values indicating better performance. For Few-shot RIR [33] andXRIR (Ours),
we evaluate in a few-shot manner by setting the number of reference RIRsKto 1, 4, and 8.

```
Method Classroom Dampened Room Hallway Complex Room
EDT C50 T60 EDT C50 T60 EDT C50 T60 EDT C50 T
Random Across Rooms 0.546 8.740 19.03 0.771 18.726 - 0.874 11.025 21.71 0.472 7.392 16.
Random Same Room 0.160 3.092 3.12 0.099 6.840 - 0.308 6.461 16.61 0.218 4.566 5.
Linear Interpolation (K=8) 0.113 2.172 4.42 0.058 4.584 - 0.088 2.127 4.55 0.124 2.848 5.
Nearest Neighbor (K=8) 0.108 1.949 2.71 0.044 3.278 - 0.068 0.990 3.02 0.091 1.936 2.
Diff-RIR [56] (K=12) 0.113 2.147 12.39 0.100 3.796 - 0.160 2.049 14.34 0.115 2.027 12.
XRIR (K=8) (Ours) 0.093 1.628 6.25 0.044 3.302 - 0.062 0.954 3.20 0.077 1.688 4.
```
Table 2.Sim-to-Real Transfer Results in Four Real Environments from the Hearing-Anything-Anywhere Dataset [56].We report
EDT Error (EDT) in seconds, C50 Error (C50) in dB, and T60 percentage error (T60). Due to noisy measurements in the dampened room,
resulting in low SNR and invalid T60 calculations on the EDC curve, we omit this metric for the dampened room.

C 50 andT 60. Our gains persists in the unseen split. In par-
ticular, our model withK= 8reducesT 60 error to around
10%, while other baselines exhibit much higher errors. This
result highlights our model’s robustness in capturing rever-
beration characteristics across different room configurations
and the ability to generalize to unseen environments with
different room acoustic properties.
The Few-Shot RIR approach from [33] does not perform
well ACOUSTICROOMS. We suspect that this is due to two
factors: i) their UNet decoder struggles to reconstruct high-
fidelity RIRs on our data, as it relies on highly compressed
fusion features; ii) their method uses binaural echoes with
co-located source and receiver positions, which fundamen-
tally differ from our setup, where reference RIRs are mea-
sured with the source and receiver at different locations.
This spatial disparity likely impacts feature relevance, lim-
iting its performance on our dataset.

#### 5.5. Sim-to-Real Transfer to Real Environments

To evaluate whether our model can also generalize to real-
world environments, we use four real rooms from the
Hearing-Anything-Anywhere Dataset [56]. We compare
our method against Diff-RIR [56], a physics-based differ-
entiable RIR rendering pipeline that utilizes 12 reference

```
RIRs per room to predict RIRs for new locations. As shown
in Table 2, our model compares favorably against all base-
lines. In partilar, despite using only 8-shot references, our
method outperforms Diff-RIR that uses 12 reference RIRs
in all acoustic metrics, demonstrating its strong generaliza-
tion capabilities. We observe that our method underper-
forms on the T60 metric compared to the Nearest Neigh-
bor baseline across all four rooms. We suspect this is be-
cause T60, as a global metric, is more sensitive to measure-
ment noise due to its aggregation of all acoustic interactions
within the room. Our learning-based method can struggle
with low SNR beyond the early parts of the waveform, as
it is trained on simulation data with higher SNR than real
room measurements. In contrast, EDT focuses on early re-
flections with high SNR, make it less noise-sensity, and C
is similarly robust due to noise smoothing in the integration
beyond the early parts. Despite this, our results demonstrate
thatXRIR’s effectiveness in adapting from simulated rooms
to real environments, successfully capturing diverse room
acoustics with fewer reference RIRs than prior methods.
```
#### 5.6. Qualitative Results

```
We present qualitative results by comparing the predicted
RIRs and acoustic maps between our modelXRIR and the
```

Figure 4.Qualitative Comparisons of RIR Predictions.We compare the performance of our method and the baselines both in simulated
(top row) and real (bottom row) environments. Room geometry, sample RIR predictions, and the corresponding error metrics are included.
XRIR shows more accurate RIR predictions in both settings.

Figure 5. Qualitative Comparisons of Acoustic Map Predictions in Two Real Environments: a Hallway and a Classroom. We
visualize the acoustics maps by computing the C50 metric at dense locations in the entire room and compare with the ground-truth acoustic
map.XRIR achieves C50 distributions that better matches the ground-truth.

baseline methods, in both simulated and real environments.
RIR Predictions.In Fig. 4, we visualize sample results
of RIR waveforms on a simulated apartment and a real room
with complex geometry. Side-by-side comparison shows
that predicted RIRs fromXRIR align more closely with the
ground-truth RIR waveforms in the early part than base-
lines. This observation is consistent with the low acoustics
metrics errors achieved by our method in the quantitative
results shown in Table 1.
Acoustics Maps.Furthermore, we compute the RIRs at
dense locations across the entire real rooms, and compute
the clarity of the predicted and ground-truth RIRs to recon-
struct acoustics maps according to the floor plans. As shown
in Fig. 5, across dense locations in these rooms, overall
XRIR achieves better C50 distribution than Diff-RIR [56]
compared to the ground-truth acoustic maps, especially at
moderate-to-low intensity regions. These qualitative results
demonstrate the effectiveness ofXRIR in accurate RIR pre-
diction in both simulation and real-world settings.

### 6. Conclusion

```
We presentedXRIR, a model designed for generalizable
RIR prediction across diverse room environments. To tackle
the cross-room RIR prediction task, we also introduced a
large-scale, realistic RIR simulation dataset, ACOUSTIC-
ROOMS, which includes diverse room categories, geome-
tries, and material properties. Results under the simulation
settings show that our framework outperforms prior meth-
ods and strong baselines in both seen and unseen environ-
ments. Furthermore, sim-to-real transfer experiments reveal
that our model, pre-trained on simulated data, effectively
generalizes to real-world settings. Future work may focus
on improving modeling techniques, such as using genera-
tive approach as proposed for sound generation [31, 52] to
achieve better performance on acoustic modeling with min-
imal reference RIRs, or dynamically choose the suitable
number of reference RIRs needed depending on the com-
plexity of the environment.
```

### References

```
[1] Byeongjoo Ahn, Karren Yang, Brian Hamilton, Jonathan
Sheaffer, Anurag Ranjan, Miguel Sarabia, Oncel Tuzel, and
Jen-Hao Rick Chang. Novel-view acoustic synthesis from
3d reconstructed rooms. arXiv preprint arXiv:2310.15130,
```
2023. 2
[2] Swapnil Bhosale, Haosen Yang, Diptesh Kanojia, Jiankang
Deng, and Xiatian Zhu. Av-gs: Learning material and ge-
ometry aware priors for novel view acoustic synthesis.arXiv
preprint arXiv:2406.08920, 2024. 2
[3] Changan Chen, Unnat Jain, Carl Schissler, Sebastia Vi-
cenc Amengual Gari, Ziad Al-Halah, Vamsi Krishna Ithapu,
Philip Robinson, and Kristen Grauman. Soundspaces:
Audio-visual navigation in 3d environments. InECCV, 2020.
2, 3, 5
[4] Changan Chen, Sagnik Majumder, Ziad Al-Halah, Ruohan
Gao, Santhosh Kumar Ramakrishnan, and Kristen Grauman.
Learning to set waypoints for audio-visual navigation. InIn-
ternational Conference on Learning Representations (ICLR),
2021. 3
[5] Changan Chen, Ruohan Gao, Paul Calamia, and Kristen
Grauman. Visual acoustic matching. InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 18858–18868, 2022. 3
[6] Changan Chen, Carl Schissler, Sanchit Garg, Philip
Kobernik, Alexander Clegg, Paul Calamia, Dhruv Batra,
Philip Robinson, and Kristen Grauman. Soundspaces 2.0: A
simulation platform for visual-acoustic learning. Advances
in Neural Information Processing Systems, 35:8896–8911,
2022. 2
[7] Changan Chen, Alexander Richard, Roman Shapovalov,
Vamsi Krishna Ithapu, Natalia Neverova, Kristen Grauman,
and Andrea Vedaldi. Novel-view acoustic synthesis. InPro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 6409–6419, 2023. 3
[8] Mingfei Chen and Eli Shlizerman. AV-cloud: Spatial
audio rendering through audio-visual cloud splatting. In
The Thirty-eighth Annual Conference on Neural Information
Processing Systems, 2024. 2, 4
[9] Ziyang Chen, Xixi Hu, and Andrew Owens. Structure from
silence: Learning scene structure from ambient sound.Con-
ference on Robot Learning (CoRL), 2021. 3
[10] Ziyang Chen, Israel D Gebru, Christian Richardt, Anurag
Kumar, William Laney, Andrew Owens, and Alexander
Richard. Real acoustic fields: An audio-visual room acous-
tics dataset and benchmark. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 21886–21896, 2024. 1, 2
[11] Sanjoy Chowdhury, Sreyan Ghosh, Dasgupta Subhrajyoti,
Anton Ratnarajah, Utkarsh Tyagi, and Dinesh Manocha. Ad-
verb: Visually guided audio dereverberation.ICCV, 2023. 3
[12] Jesper Haahr Christensen, Sascha Hornauer, and X Yu Stella.
Batvision: Learning to see 3d spatial layout with two ears. In
ICRA. IEEE, 2020. 3
[13] Elke Deckers, Onur Atak, Laurens Coox, Roberto D’Amico,
Hendrik Devriendt, Stijn Jonckheere, Kunmo Koo, Bert
Pluymers, Dirk Vandepitte, and Wim Desmet. The wave

```
based method: An overview of 15 years of research.Wave
Motion, 51(4):550–565, 2014. 2
[14] Alexey Dosovitskiy. An image is worth 16x16 words:
Transformers for image recognition at scale.arXiv preprint
arXiv:2010.11929, 2020. 4
[15] Chuang Gan, Yiwei Zhang, Jiajun Wu, Boqing Gong, and
Joshua B Tenenbaum. Look, listen, and act: Towards audio-
visual embodied navigation. InICRA, 2020. 3
[16] Ruohan Gao and Kristen Grauman. 2.5d visual sound. In
Conference on Computer Vision and Pattern Recognition
(CVPR), 2019. 3
[17] Ruohan Gao, Changan Chen, Ziad Al-Halah, Carl Schissler,
and Kristen Grauman. Visualechoes: Spatial visual repre-
sentation learning through echolocation. InEuropean Con-
ference on Computer Vision (ECCV), 2020. 3
[18] Ruohan Gao, Hao Li, Gokul Dharan, Zhuzhu Wang, Cheng-
shu Li, Fei Xia, Silvio Savarese, Li Fei-Fei, and Jiajun Wu.
Sonicverse: A multisensory simulation platform for training
household agents that see and hear. InInternational Confer-
ence on Robotics and Automation (ICRA),, 2023. 3
[19] Rishabh Garg, Ruohan Gao, and Kristen Grauman.
Geometry-aware multi-task learning for binaural audio gen-
eration from video. InBritish Machine Vision Conference
(BMVC), 2021. 3
[20] Rishabh Garg, Ruohan Gao, and Kristen Grauman. Visually-
guided audio spatialization in video with geometry-aware
multi-task learning. InInternational Journal of Computer
Vision (IJCV), 2023. 3
[21] D. Griffin and Jae Lim. Signal estimation from modified
short-time fourier transform.IEEE Transactions on Acous-
tics, Speech, and Signal Processing, 32(2):236–243, 1984.
5
[22] Reinhold Haeb-Umbach, Jahn Heymann, Lukas Drude,
Shinji Watanabe, Marc Delcroix, and Tomohiro Nakatani.
Far-field automatic speech recognition. Proceedings of the
IEEE, 109(2):124–148, 2020. 2
[23] Brian Hamilton and Craig J. Webb. Room acoustics mod-
elling using gpu-accelerated finite difference and finite vol-
ume methods on a face-centered cubic grid. 2013. 5
[24] Zitong Lan, Chenhao Zheng, Zhiwei Zheng, and Mingmin
Zhao. Acoustic volume rendering for neural impulse re-
sponse fields. arXiv preprint arXiv:2411.06307, 2024. 2,
4
[25] Tobias Lentz, Dirk Schroder, Michael Vorl ̈ ̈ander, and Ingo
Assenmacher. Virtual reality system with integrated sound
field simulation and reproduction.EURASIP journal on ad-
vances in signal processing, 2007:1–19, 2007. 2
[26] Dingzeyu Li, Timothy R Langlois, and Changxi Zheng.
Scene-aware audio for 360 videos. ACM Transactions on
Graphics (TOG), 37(4):1–12, 2018. 3
[27] Tingle Li, Renhao Wang, Po-Yao Huang, Andrew Owens,
and Gopala Anumanchipalli. Self-supervised audio-visual
soundscape stylization. InECCV, 2024. 3
[28] Susan Liang, Chao Huang, Yapeng Tian, Anurag Kumar, and
Chenliang Xu. Av-nerf: Learning neural fields for real-world
audio-visual scene synthesis.Advances in Neural Informa-
tion Processing Systems, 36:37472–37490, 2023. 2
```

[29] Susan Liang, Chao Huang, Yapeng Tian, Anurag Kumar,
and Chenliang Xu. Neural acoustic context field: Render-
ing realistic room impulse response with neural fields.arXiv
preprint arXiv:2309.15977, 2023. 1
[30] Xiulong Liu, Sudipta Paul, Moitreya Chatterjee, and Anoop
Cherian. Caven: An embodied conversational agent for ef-
ficient audio-visual navigation in noisy environments. Pro-
ceedings of the AAAI Conference on Artificial Intelligence,
38(4):3765–3773, 2024. 3
[31] Xiulong Liu, Kun Su, and Eli Shlizerman. Tell what you
hear from what you see - video to audio generation through
text. InAdvances in Neural Information Processing Systems,
pages 101337–101366. Curran Associates, Inc., 2024. 8
[32] Andrew Luo, Yilun Du, Michael Tarr, Josh Tenenbaum, An-
tonio Torralba, and Chuang Gan. Learning neural acoustic
fields.Advances in Neural Information Processing Systems,
35:3165–3177, 2022. 1, 2, 3, 6, 4
[33] Sagnik Majumder, Changan Chen, Ziad Al-Halah, and Kris-
ten Grauman. Few-shot audio-visual learning of environ-
ment acoustics.Advances in Neural Information Processing
Systems, 35:2522–2536, 2022. 2, 5, 6, 7, 1
[34] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[35] Pedro Morgado, Nuno Nvasconcelos, Timothy Langlois, and
Oliver Wang. Self-supervised generation of spatial audio for
360 video. Advances in neural information processing sys-
tems, 2018. 3
[36] Brady Peters. Integrating acoustic simulation in architectural
design workflows: the fabpod meeting room prototype.Sim-
ulation, 91(9):787–808, 2015. 2
[37] Anton Ratnarajah, Zhenyu Tang, and Dinesh Manocha. Ir-
gan: Room impulse response generator for far-field speech
recognition.arXiv preprint arXiv:2010.13219, 2020. 2
[38] Anton Ratnarajah, Zhenyu Tang, and Dinesh Manocha. IR-
GAN: Room Impulse Response Generator for Far-Field
Speech Recognition. InProc. Interspeech 2021, pages 286–
290, 2021. 1
[39] Anton Ratnarajah, Zhenyu Tang, Rohith Aralikatti, and Di-
nesh Manocha. Mesh2ir: Neural acoustic impulse response
generator for complex 3d scenes. InProceedings of the 30th
ACM International Conference on Multimedia, pages 924–
933, 2022. 2
[40] Anton Ratnarajah, Shi-Xiong Zhang, Meng Yu, Zhenyu
Tang, Dinesh Manocha, and Dong Yu. Fast-rir: Fast neural
diffuse room impulse response generator. InICASSP 2022-
2022 IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP), pages 571–575. IEEE,

2022. 2
[41] Anton Ratnarajah, Sreyan Ghosh, Sonal Kumar, Purva
Chiniya, and Dinesh Manocha. Av-rir: Audio-visual
room impulse response estimation. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 27164–27175, 2024. 2
[42] W H Reed and T R Hill. Triangular mesh methods for the
neutron transport equation. 1973. 2, 6

```
[43] Alexander Richard, Peter Dodds, and Vamsi Krishna Ithapu.
Deep impulse responses: Estimating and parameterizing fil-
ters with deep networks. InICASSP, pages 3209–3213.
IEEE, 2022. 1, 2
[44] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets,
Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia
Liu, Vladlen Koltun, Jitendra Malik, et al. Habitat: A
platform for embodied ai research. InProceedings of
the IEEE/CVF international conference on computer vision,
pages 9339–9347, 2019. 4
[45] Robin Scheibler, Eric Bezzam, and Ivan Dokmani ́c. Py-
roomacoustics: A python package for audio room simula-
tion and array processing algorithms. In2018 IEEE interna-
tional conference on acoustics, speech and signal processing
(ICASSP), pages 351–355. IEEE, 2018. 2
[46] Carl Schissler, Gregor M ̈uckl, and Paul Calamia. Fast
diffraction pathfinding for dynamic sound propagation. 40
(4), 2021. 2
[47] Manfred R Schroeder. New method of measuring reverbera-
tion time.The Journal of the Acoustical Society of America,
37(6Supplement):1187–1188, 1965. 6
[48] Nikhil Singh, Jeff Mentch, Jerry Ng, Matthew Beveridge,
and Iddo Drori. Image2reverb: Cross-modal reverb impulse
response synthesis. InProceedings of the IEEE/CVF Inter-
national Conference on Computer Vision, pages 286–295,
```
2021. 2
[49] Vincent Sitzmann, Julien Martel, Alexander Bergman, David
Lindell, and Gordon Wetzstein. Implicit neural representa-
tions with periodic activation functions.Advances in neural
information processing systems, 33:7462–7473, 2020. 2
[50] Arjun Somayazulu, Changan Chen, and Kristen Grauman.
Self-supervised visual acoustic matching.Advances in Neu-
ral Information Processing Systems, 36, 2024. 3
[51] Kun Su, Mingfei Chen, and Eli Shlizerman. Inras: Implicit
neural representation for audio scenes.Advances in Neural
Information Processing Systems, 35:8144–8158, 2022. 1, 2,
3, 4, 6
[52] Kun Su, Xiulong Liu, and Eli Shlizerman. From vision to
audio and beyond: A unified model for audio-visual repre-
sentation and generation.arXiv preprint arXiv:2409.19132,
2024. 8
[53] Zhenyu Tang, Rohith Aralikatti, Anton Jeran Ratnarajah, and
Dinesh Manocha. Gwa: A large high-quality acoustic dataset
for audio processing. InACM SIGGRAPH 2022 Conference
Proceedings, pages 1–9, 2022. 2, 5
[54] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia
Polosukhin. Attention is all you need. InAdvances in Neu-
ral Information Processing Systems. Curran Associates, Inc.,
2017. 4, 5
[55] Mason Wang, Samuel Clarke, Jui-Hsien Wang, Ruohan Gao,
and Jiajun Wu. Soundcam: a dataset for finding humans us-
ing room acoustics. Advances in Neural Information Pro-
cessing Systems, 36, 2024. 3
[56] Mason Long Wang, Ryosuke Sawata, Samuel Clarke, Ruo-
han Gao, Shangzhe Wu, and Jiajun Wu. Hearing anything
anywhere. InProceedings of the IEEE/CVF Conference


on Computer Vision and Pattern Recognition, pages 11790–
11799, 2024. 2, 6, 7, 8


## Hearing Anywhere in Any Environment

## Supplementary Material

### 7. Summary of Supplementary Materials

```
In our supplementary materials, we provide:
```
1. Supplementary video. 8
2. Additional details including explanation of reference
    RIR setup, implementations ofXRIR and baselines as
    well as experiments setup, see Section 9.
3. More details about ACOUSTICROOMS, see Section 10.
4. Single-room RIR prediction results compared to prior
works, see Section 11.
5. Additional experiments and ablation studies of the xRIR,
see Section 12.
6. Additional qualitative samples, RIR prediction compari-
son 13.

### 8. Supplementary Video

In the supplementary video, we provide a brief summary
of our work and qualitative samples of audio rendered with
different predicted RIRs in simulation as well as real envi-
ronments. For best perceptual experience, please turn the
Audio ON and use headphone when watching.
For demos in the supplementary video that demonstrate
audio rendering along trajectories in real scenes, we con-
volve our predicted single-channel Room Impulse Response
(RIR) at each point in the trajectory with a Head-Related
Impulse Response (HRIR) from a predefined Head-Related
Transfer Function (HRTF). This process yields binaural
RIRs, which capture spatial effects. We then convolve these
binaural RIRs with the source audio to obtain the binaural
audio along these trajectories.

### 9. Additional Details

Further explanation on reference RIR setup: Our ref-
erence RIRs setup is practically useful. Model trained on
this setup is able to predict new RIRs without anyrefer-
ence RIRs re-measurementwhen either source or receiver
freely moves in the target room. Our method is capable of
addressing both scenarios below:
i)Fixed reference receiver, multiple sources: Measure
reference RIRs between a fixed receiver and sources at var-
ious locations, then the model predicts the RIR for the fixed
receiver and a new source location (ACOUSTICROOMS
setup).
ii)Fixed reference source, multiple receivers: Measure
reference RIRs between a fixed source and receivers at var-
ious locations, then the model predicts the RIR for the
fixed source and a new receiver location (HearAnythin-
gAnywhere setup).

```
A model trained on i) can be directly applied to ii) by
switching source and receiver subscripts (due to symmetry
of wave equation for single-channel RIR), as shown in our
sim-to-real transfer experiment. We adopt scenario i) in our
task, where the receiver always matches the reference re-
ceiver location.
XRIR:ForXRIR, we implement a Vision Transformer
blockFvtwith 6 multi-head attention layers (8 heads, hid-
den size 512). For panorama depth map, the center pixel
corresponds to the receiver location. Two spherical angles
maps are initialized for equirectangular projection from the
depth map to 3D coordinates map. In the vision transformer
module, the 3D coordinates map is divided into 16 × 32
patches, resulting in all reflection-based features such asgr,rf
andgs,rfof dimension 256 × 512. Direct path features are
calculated using sinusoidal positional encoding on each 3D
coordinate with 20 frequency bins, and are then projected
into 256-dimensional vectors via MLP. Similarly, the time
basis vectorTbis calculated by sinusoidal positional encod-
ing with 10 frequency bins for each time index, where the
length ofTbis same as the length of spectrogram, 310. Be-
fore performing weight combination, we further preprocess
the reference RIRs by time-shifting them based on the dis-
tance difference between the target and reference source-
receiver pairs, divided by the speed of sound. For loss cal-
culation, we setλ= 0. 01 to balance the STFT loss and the
energy decay loss.
Few-Shot RIR:Unlike the approach and the problem setup
in [33], which use binaural echoes where the source and
receiver are co-located to predict a target binaural RIR,
we use reference RIRs measured with the source at differ-
ent locations from the receiver location to predict single-
channel RIR at a target source. This is very important since
the echo input used by [33] are infeasible to obtain un-
der the single-channel RIR scenario, because it is not rea-
sonable in physics to co-locate source and receiver at the
same location to measure the single-channel RIR. In addi-
tion, we also omit the RGB image for the visual input and
use only depth maps as inputs to the vision branch of the
Few-shot RIR model, due to the weak correspondence be-
tween room semantics and material properties in ACOUSTI-
CROOMSdataset. We also emphasize that we use panorama
depth images captured from each reference source loca-
tion instead of egocentric depth images as the depth inputs.
For all depth observations, we rendered at a resolution of
128 × 256. Except for the adaptations above, all other im-
plementation details follow the Few-Shot RIR model [33].
Diff-RIR:We use their released Github code and model
```

checkpoints to perform evaluations on all rooms in the
Hearing-Anything-Anywhere Dataset [56]. We strictly fol-
lowed the inference and evaluation settings in the paper, and
obtained the same results in terms of the metric errors re-
ported in the paper (Mag and Env error metrics) to make
sure there are no implementation issues. And then we eval-
uate their inference results on our three acoustic metrics
which are more related to perceptual quality of the RIRs:
EDT error, C50 error and T60 error.
Experiments Setup: For cross-room RIR prediction ex-
periments on ACOUSTICROOMS, we manually select 10
sources in each simulated room as candidate reference
sources to make sure that their spatial locations are evenly
distributed within the scene as much as possible. For the
seen setting, we split training and test set within each room
by receivers, where the RIRs of 90% of receivers belong to
training split and remaining 10% belong to test split. For
the unseen setting, we split the data by rooms. For each
room category, we use 90% of rooms for training and 10%
rooms for testing. During both training and testing, we ran-
domly selectK= 1, 4 , 8 reference RIRs from these candi-
date sources. And for eachK, we train a separate model on
the dataset.
For experiments on the Hearing-Anything-Anywhere
Dataset [56], we sample referenceK= 8RIRs from their
selected 12 reference RIRs in each room. In this real-world
dataset, the source has specific directivity patterns which
are not captured in our pretrained model using ACOUS-
TICROOMS. Therefore, we further finetune the pretrained
XRIR model on these 12 selected RIRs to make sure a fair
comparison with Diff-RIR [56]. Specifically, in each itera-
tion during finetuning, we randomly sample 8 of 12 as refer-
ence RIRs and predict a target RIR sampled from remaining
4 RIRs. We also use the same validation set as Diff-RIR to
select the model checkpoint for testing. We select the one
with lowest validation loss to evaluate on the test split of the
dataset. It is note-worthy that even thoughXRIR is fine-
tuned, compared to the training time of Diff-RIR on these
few shot samples (6 hours / scene),XRIR converges much
faster than Diff-RIR, with a matter of minutes. This helps
us to quickly perform sim-to-real transfer across different
environments efficiently.

### 10. Dataset Details

In this section, we present details about our large-scale
simulated RIR dataset, ACOUSTICROOMS, used for cross-
room RIR prediction task. ACOUSTICROOMScontains
260 rooms from 10 different categories, simulating a to-
tal of 30,000 RIRs from different source-receiver pairs.
ACOUSTICROOMSfeatures professional room architecture
designs of high quality, covering a wide range of room cat-
egories, including: apartment, auditorium, bathroom, bed-

```
room, cafe, listening room, living room, meeting room, of-
fice and restaurant, as shown in Figure 6. The area of rooms
ranges from 20 m^3 to 1000 m^3 , with diverse range of sizes
and geometries. ACOUSTICROOMSuses a realistic and
commercial acoustics simulation platform, Treble, to per-
form single-channel RIR simulation. The platform supports
a wide variety of simulation methods along with specific
settings. To obtain more realistic RIR data as well as simu-
late large-scale data, we adopt the hybrid-based simulation.
At low frequency bands, an advanced wave-based method
is used to capture more subtle wave interaction effects such
as diffraction and resonance. At high frequency bands, we
use geometric-based simulation that combines two simula-
tion techniques: image-source technique and stochastic ray-
tracing technique.
Source Receiver Placement:To set up the simulation for
each room, we first choose the sources and receivers and
place them at different locations. For sources, we use omni-
directional source devices without particular directivity pat-
terns since our goal is not to overfit the model to a particular
device pattern. Similarly, for receivers, we use monaural re-
ceivers such that they do not model specific HRTF patterns.
Depending on the size of rooms, we place 10 to 100 sources
and 25 to 100 receivers per room, to ensure they sufficiently
cover the whole area of the rooms. To determine the loca-
tion of each source and receiver, we apply a set of place-
ment rules to avoid interference among devices and room
surfaces when the distance becomes too small to cause issue
in the simulation quality. We require that: i) Sources should
be at least 0. 5 maway from each room surface, 1. 0 maway
from other sources and 1. 0 maway from receivers. ii) Re-
ceivers should be at least 0. 5 maway from room surface and
at least 0. 5 maway from the sources. Given these rules, we
apply a point-picking algorithm to randomly sample valid
source and receiver locations within each room at different
height level from 0. 5 mto 2. 5 m.
Material Assignment:Once the source and receivers are
determined, we assign materials to room surfaces by associ-
ating their semantic labels with particular material category.
Treble platform provides a large-scale material database
with 332 specific materials from 11 material categories,
with each category containing 30 different material coef-
ficients on average. We define the mapping between each
semantic labels of room surfaces and the 11 material cate-
gories. In each room, each semantic label of a particular sur-
face gets randomly mapped to one of the specific material
with a set of acoustics coefficients under the material cate-
gory. Different from existing RIR datasets [3, 6, 53], this
random assignment ensures enough diversity in the mate-
rial properties of the room surfaces. Even two rooms share
similar geometries and semantic objects, they could have
very different acoustics behavior due to differences in their
specific acoustics coefficients of materials.
```

```
Figure 6. A visualization of different room categories in ACOUSTICROOMS.
```
```
Figure 7. An overview of procedures to simulate RIRs in ACOUSTICROOMS.
```
Simulation Setup:Once sources and receivers are in place
and room materials get assigned, we set up the simulation
by specifying the hybrid mode to split the geometric-based
and wave-based method. We choose the crossover fre-
quency to befcross= 710Hz between two methods such that
wave-based effects could be sufficiently captured across dif-
ferent rooms and objects of different size. For geometric-
based simulation, we use image source method up to re-
flection order of 4 with 50 krays emitting from the source.
For refection orders higher than 4, we apply stochastic ray-
tracing method with 5000 rays. With this configuration, we
simulate the RIR of each room until 60 dBenergy decay
to ensure all possible acoustics effects are sufficiently cap-
tured.

### 11. Comparison on Single-Room RIR Predic-

### tion

```
Adaptation ofXRIR:AlthoughXRIR focuses on solving
cross-room RIR prediction task, it could be easily adapted
to single-room RIR prediction task as well. By removing all
the components related to reference RIRs, we extract the ge-
ometric and spatial features related to only target source and
receiver positions. We then follow [51] to use their implicit
neural decoder to perform RIR waveform synthesis. Specif-
ically, given a time basis vectorB, and the outputs from
Geometric feature extractorgdir,gr,rfandgs,rf, we learn a
implicit neural mapping function to synthesize time domain
RIR waveform from the outer product betweenBwith the
three features:Aˆt=Finr(gdirBT,gr,rfBT,gs,rfBT). We
train the adapted model with same loss function as in [51],
the multi-resolution STFT loss combined with waveform
L2 loss.
```

```
Model
Apartment 1 Apartment 2 FRL Apartment 2
EDT↓ C50↓ T60↓ EDT↓ C50↓ T60↓ EDT↓ C50↓ T60↓
NAF 0.077 0.426 7.508 0.066 0.453 7.925 0.088 0.420 6.
INRAS 0.027 1.036 6.514 0.025 0.843 5.816 0.022 0.634 2.
xRIR 0.026 1.000 6.192 0.031 0.932 5.755 0.021 0.587 1.
```
```
Model Room 2 FRL Apartment 4 Office 4 Mean
EDT↓ C50↓ T60↓ EDT↓ C50↓ T60↓ EDT↓ C50↓ T60↓ EDT↓ C50↓ T60↓
NAF 0.056 0.407 4.969 0.085 0.421 7.475 0.081 0.337 6.760 0.076 0.411 6.
INRAS 0.020 0.555 1.990 0.022 0.625 2.145 0.014 0.610 3.251 0.022 0.717 3.
xRIR 0.019 0.541 1.910 0.021 0.561 2.140 0.013 0.502 2.767 0.022 0.687 3.
```
```
Table 3. Performance comparison on single-room RIR prediction task on six rooms in SoundSpaces 1.0 - Replica dataset in terms of EDT
(s), C50 (dB), and T60 (%) error metrics.
```
Experiment Setup: In single-room RIR prediction task,
the goal is to the fit scene acoustic with dense RIR obser-
vations. Therefore, we use the standard dense RIR dataset,
SoundSpaces 1.0 Replica, to perform the experiment. Fol-
lowing prior works [32, 51], we use the six scenes from
Replica. But instead of using binaural RIR data, we use
single-channel RIR data by extracting the first channel of
ambisonic RIR data of these scenes. For each scene, we
split the RIR data into training and test set with a ratio of
9:1. We cut the RIR to the maximum length of 8000 sam-
ples (0.363s) at sampling rate 22,050Hz for all six rooms.
And for panorama image at each receiver location, we ren-
der it by setting the orientation to 0 in the habitat simula-
tor [44].

Baselines:We compareXRIR with two prior works of the
state-of-the-art performance on the dataset, NAF [32] and
INRAS [51]. For both methods, we remove the orientation
conditioning vector to adjust for single-channel RIR pre-
diction, while keeping the remaining implementations the
same. For evaluation, we use the same metrics in the cross-
room RIR prediction task for comparison.

Quantatitive Results:We report individual results for each
of six scenes and their average results and show them in Ta-
ble 3. As could be seen, when compared to NAF,XRIR
outperforms on both EDT and T60 metrics, while slightly
underperforming in C50 error. When compared to INRAS,
XRIR outperform INRAS on 5 out of 6 scenes, except
“Apartment 2”. The reason is due to the fact that the mesh
of “Apartment 2” shows significant amount of holes in a re-
gion, which leads to degraded quality of panorama depth in-
puts. While, INRAS does not suffer from this degradation
since the method samples mesh points instead of render-
ing images. Overall,XRIR, when adapted to single-room
RIR prediction task, shows on par or even better perfor-
mance when compared to these prior arts, demonstrating
the effectiveness of Geometric Feature Extractor to learn the

```
scene acoustics from local geometric observations (around
receiver).
```
```
Method Classroom Dampened Hallway Complex
MAG ENV MAG ENV MAG ENV MAG ENV
Random Across 1.98 4.45 3.571 7.573 3.415 7.571 1.762 6.
Random Same 0.710 2.182 0.213 1.635 1.104 6.582 0.685 2.
Linear Interp 0.725 1.890 0.110 0.908 1.082 5.566 0.637 2.
Nearest Neigh 0.600 2.003 0.108 0.916 0.793 5.589 0.542 2.
Diff-RIR (K=12) 0.486 1.8260.085 0.8830.7245.173 0.4422.
xRIR (K=8) 0.456 1.8240.093 0.8920.7185.320 0.4662.
```
```
Table 4. Sim-to-Real Transfer Resultsusing MAG and ENV
metrics on HearingAnythingAnywhere dataset.
```
```
Method EDT C50 T
XRIR w.o Reference RIRs 0.166 3.925 32.
XRIR w.o Direct Path Module 0.061 1.596 12.
XRIR w.o Reflection Module 0.059 1.498 11.
XRIR (full) 0.055 1.457 10.
```
```
Table 5. Ablation Study: Comparison of different ablatedXRIR
components on ACOUSTICROOMSwith unseen splits. We report
EDT (s), C50 (dB), and T60 (%) error metrics
```
### 12. Additional Experiments and Ablation

### Studies

```
Additional Comparisons on HearingAnythingAny-
where dataset: We use MAG from [8] and ENV from
[24] to further evaluate waveform similarity. For fair
comparison, we evaluate our model on the first 0.435s of
RIRs due to model length constraints. As shown in Table
4, our method performs similarly to DiffRIR on these
metrics but outperforms in perceptual metrics like EDT and
C50 (Table 2 main paper), which better reflect perceptual
```

```
Figure 8. Additional qualitative comparisons on RIR waveform predictions in ACOUSTICROOMS.
```
```
Figure 9. Additional qualitative comparisons on RIR waveform predictions on the Hearing-Anything-Anywhere Dataset.
```
quality of rendered RIRs.

Ablation Studies:
We perform ablation studies on the components ofXRIR

```
by considering the followings:
```
- XRIR w.o Reference RIRs: This ablated variant is the
    same as our model’s adaptation to single-room RIR pre-


```
diction task. Since there are no reference RIRs as inputs,
the model utilizes the implicit neural function to synthe-
size the target RIR.
```
- XRIR w.o Direct Path Module: By removing the Direct
    Path Module, the model only considers the relationship
    between sources / receivers and room geometry informa-
    tion, without computing the direct path features.
- XRIR w.o Reflection Module: By removing the Reflec-
    tion Module, the model only considers the spatial rela-
    tionship between sources and receivers without taking lo-
    cal geometry information into account.
       For the above ablation variants, we perform experiments
on ACOUSTICROOMSunder unseen settings. We set the
number of reference RIRsK = 8for models that take
reference RIRs as inputs. As shown in Table 5, our full
model outperforms all ablated variants across all metrics.
It is note-worthy that without providing reference RIR as
inputs to the model, the model is not able to synthesize rea-
sonable RIRs by just relying on geometric and positional
inputs. Also, removing either Direct Path module or Re-
flection module will lead to degraded performance across
all acoustics metrics, demonstrating the importance of cap-
turing full spatial and geometric information for accurate
RIR predictions.
Furthermore, we study the importance of finetuning
as well as the pretraining on our simulation dataset. In
general, we find that simulation data helps the model
capture general acoustic properties, such as geometry and
material effects. Finetuning on just 12 real samples allows
the model to adapt to specific factors like the source’s
directivity, improving EDT and C50 metrics compared to
training from scratch or without finetuning, as shown in
Table 6.

```
Setting EDT (s) C50 (dB) T60 (%)
Scratch 0.322 4.322 7.
Pretrained 0.204 3.427 4.
Finetuned 0.092 1.614 6.
Table 6. Importance of finetuning and pretraining on HearingAny-
thingAnywhere dataset (classroom).
```
In addition, to study the impact of scale of ACOUSTIC-
ROOMSon the performance of model, we retrain our model
on different number of rooms using xRIR (8-shot) in the
unseen setting, while keeping the test split the same. As
shown in Table 7, performance improves with more data
but diminishes as room count increases.

### 13. Additional Qualitative Samples

We provide additional qualitative results of comparisons
between our modelXRIR and the baseline methods on

```
Rooms EDT (s) C50 (dB) T60 (%)
65 0.088 1.813 13.
130 0.062 1.578 11.
260 0.055 1.457 10.
```
```
Table 7. Impact of data scale on model performance.
```
```
the predicted RIRs on both the simulation dataset and real
dataset.
As shown in Figure 8 and 9, we visualize predicted
RIR waveforms versus ground truth RIRs on three simula-
tion environments (apartment, restaurant and bathroom) in
ACOUSTICROOMSas well as two real environments (hall-
way and classroom) in the Hearing-Anything-Anywhere
dataset. At same location in these environments, RIRs pre-
dicted byXRIR align more closely with the ground truth
RIRs, demonstrating the effectiveness ofXRIR in RIR pre-
dictions under both simulated and real settings.
```

