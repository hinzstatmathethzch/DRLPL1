# **Description of data structures used**

### Struct `ModelRewrite`

Represents the topology of the neural network that arises from an input model and a layer index `ell` by writing the output as a function of the layer parameters.

```julia
struct ModelRewrite
    n0::Int
    n::Array{Int,1}
    nNoDuplicate::Array{Int,1}
    L::Int
    W2::Array{Array{Float64,2},1}
    c::Array{Array{Float64,1},1}
end
```

The struct members have the following purpose:

- `n0`: The input dimension of the rewritten feed-forward neural network
- `n`: An array containing the layer widths of the rewritten network
- `nNoDuplicate`: The input model also has a member "`nNoDuplicate`" and the rewritten network inherits this information appropriately.
- `L`: The number of layers of the rewritten network.
- `W2`: The weight matrices with index greater than `ell` of the original input model  appear starting at layer 2 in the rewrite. The layer 1 weight matrix of the rewritten model depends on the transformed predictors (output of layer `ell-1` activations in the original model ) and is therefore specific to the `DataBranch`.
- `c`: The bias vectors of the rewritten network. Note that the final layer using the absolute value function has an additional bias induced by the response value `y` specific to the `DataBranch`.

### Struct `DataBranch`

Provides additional properties for a `ModelRewrite` which are specific to one observation of the predictor and response data.

```julia
mutable struct DataBranch
    # variables that need a single copy per data point for training
	duplicateGroup::Int;
	isRepresentative::Bool
	s::Array{Array{Bool,1},1}
	grad::Array{Float64,1}
	critical::Array{Tuple{Int,Int},1}
	W1::Matrix{Float64}
	y::Array{Float64,1}
end
```

The struct members have the following purpose:

- `duplicateGroup`: A value of `0` indicates that this data branch is not contained in one of the arrays of `duplicateGroups` of the corresponding `TrainerState`. Otherwise, a non-zero value corresponds to the index of the array in `duplicateGroups` where this `DataBranch` is referred to by its index in the array `branches` of the `TrainerState`.
- `isRepresentative`: Is set to `true` if this data branch has a tuple element in its `critical` array with a layer index (first element of tuple) smaller than the number `L` of layers in the `ModelRewrite`.
- `s`: An array of boolean arrays. The indices of the outer and inner array correspond to layer number and neuron index within that layer. The boolean values specify if the corresponding neuron is active (`1`)  or inactive (`0`).
- `grad`: The gradient currently computed for this data branch. Due structure of the L1-loss for the regression training of neural networks, the gradients `grad` of multiple `DataBranch`es can be summed up to obtain the overall gradient with respect to the arguments of the rewritten neural network, i.e. with respect to the layer `ell` weight and bias parameters of the original model.
- `critical`: An array of tuples containing the layer index and neuron index within that layer of neurons which are currently critical (i.e. for which the algorithm has computed axis directions in the `Apseudo` matrix)
- `W1`: The matrix of transformed predictors (see `W2` in `ModelRewrite` above) for the first layer of the rewritten model when this observation of the predictor data is used.
- `y`: The response value of the corresponding observation.

### Struct `TrainerState`

```julia
mutable struct TrainerState
    duplicateGroups::Array{Array{Int,1},1}
	criticalDuplicateGroups::Array{Bool,1}
    rewrite::ModelRewrite
    Apseudo::Array{Float64,2}
    branches::Array{DataBranch,1}
    pos::Array{Float64,1} # position for theta_l
    critical::Array{Tuple{Int,Int,Int},1}
    gradient::Array{Float64,1}
    direction::Array{Float64,1}
end
```

The struct members have the following purpose:

- `duplicateGroups`: An Array of `Int`-Arrays; each such `Int`-Array groups together branch indices which have the same transformed input data, i.e. the same matrix `W1`.
- `criticalDuplicateGroups`: A boolean array specifying whether `critical` contains a tuple corresponding to neuron in a data branch within a duplicate group (i.e. element of `duplicateGroups`) with layer index smaller than `rewrite.L` (because in that layer the hyperplanes induced by different `DataBranch` elements in the same duplicate group can still differ because of the potentially different response variable `y`)
- `rewrite`: A model rewrite, see above.
- `Apseudo`: The peudo-inverse of the matrix with columns equal to the normal vectors of the hyperplanes corresponding to the critical neurons with indices in `critical` below.
- `branches`: A vector of `DataBranch`es, one for every observation involved in the training process.
- `pos`: The current position in the input space of the rewritten neural network, i.e. the current weight and bias vector values in the layer `ell` of the original model.
- `critical` An array of the indices of critical neurons that correspond to the matrix `Apseudo`. The indices are a three-component tuple, where the first component corresponds to the data branch index in `branches` and the other two indices are as in the `critical` property of a `DataBranch`. This copy is needed for performance reasons and should always be in sync with the `critical` properties of the data branches.
- `gradient`: The gradient of the resulting function (sum of `gradient` properties of the individual branches in `branches`). This copy is needed for performance reasons.
- `direction`: The direction used in the algorithm, it is derived from the `gradient` such that the critical hyperplanes are not left.