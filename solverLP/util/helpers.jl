include("../../models/model.jl")
using LinearAlgebra

####################
# Define the neural network itself 
# and similar functions
####################
htilde(m::Model,layerNum,input)= m.W[layerNum]*input+m.b[layerNum]
ReLU(x)=max(0,x)
h(m::Model,l,x)=ReLU.(htilde(m,l,x))
function layerOutput(m::Model,x::Array{Float64,1},l::Int)
	tmp=x
	for i = 1:l
		tmp=h(m,i,tmp)
	end
	return tmp
end
function constructTMatrix(rep::Int,x::Array{Float64,1})
	k=size(x,1)
	W=zeros(Float64,rep,rep*(k+1))
	for i = 1:rep
		for j = 1:k
			W[i,(i-1)*(k+1)+j]=x[j]
		end
		W[i,(i)*(k+1)]=1.0
	end
	return W
end
struct ModelRewrite
	n0::Int
	n::Array{Int,1}
	nNoDuplicate::Array{Int,1}
	L::Int
	W2::Array{Array{Float64,2},1}
	bNeedAdjustEnd::Array{Array{Float64,1},1}
	function ModelRewrite(model::Model, ell::Int)
		@assert(ell<=model.L+1)
		@assert(ell>=1)
		nPrev= (ell==1 ? model.n0 : model.n[ell-1])
		nEll=(ell<=model.L ? model.n[ell] : 1)
		n0=nEll*(nPrev+1)
		# Note that to we hard-code the convention that the last activation function is
		# the absolute value function (and not represented by two ReLU units). Hence we
		# do not need L=model.L-ell+2 layers but only L=model.L-ell+1.
		n=Array{Int,1}( (ell<=model.L ? [model.n[(ell):model.L]...,1] : [1]))
		nNoDuplicate=Array{Int,1}( (ell<=model.L ? [model.nNoDuplicate[(ell):model.L]...,1] : [1]))
		L=model.L-ell+2
		W2=Array{Matrix,1}((model.W[ell+1:end]))
		nlast=(ell <=model.L ? model.n[ell] : 1)
		bNeedAdjustEnd=Array{Array{Float64,1},1}([repeat([0],outer=nlast),(model.b[ell+1:end])...])
		new(n0,n,nNoDuplicate,L,W2,bNeedAdjustEnd)
	end
end
function sigGrad(m::ModelRewrite,W1::Matrix, y::Array{Float64,1}, pos::Array{Float64,1})
	s=Array{Array{Bool,1},1}(undef,m.L)
s=Array{Array{Bool,1},1}(undef,m.L)
grad=Matrix{Float64}(I,m.n0,m.n0)
tmp=W1*pos
#
for l = 1:m.L-1
	s[l]=Array{Bool,1}(undef,m.n[l])
	for j = 1:m.nNoDuplicate[l]
		s[l][j]=(tmp[j]>0)
	end
	diff=m.n[l]-m.nNoDuplicate[l]
	for j = (m.nNoDuplicate[l]+1):m.n[l]
		s[l][j]=!s[l][j-diff]
	end
	# W2 offset shifted, hence subtract 1
	if l>1
		grad=s[l].*(m.W2[l-1]*grad)
	else
	grad=s[l].*(W1*grad)
	end
	tmp=ReLU.(tmp)
	tmp=m.W2[l+1-1]*tmp+m.bNeedAdjustEnd[l+1]
end
#
# last layer
tmp -=y
s[m.L]=Array{Bool,1}(undef,m.n[m.L])
for j = 1:m.nNoDuplicate[m.L]
	s[m.L][j]=(tmp[j]>0)
end
dif=m.n[m.L]-m.nNoDuplicate[m.L]
for j = (m.nNoDuplicate[m.L]+1):m.n[m.L]
	s[m.L][j]=!s[m.L][j-dif]
end
if m.L>1
	grad=m.W2[m.L-1]*grad
else
	grad=W1*grad
end
if tmp[1]<0
	grad=-grad
end
	return (s,grad[1,:])
end
mutable struct DataBranch
	# variables that need a single copy per data point for training
	duplicateRepresentative::Int;
	s::Array{Array{Bool,1},1}
	grad::Array{Float64,1}
	critical::Array{Tuple{Int,Int},1}
	W1::Matrix{Float64}
	y::Array{Float64,1}
	function DataBranch(m::ModelRewrite, W1::Matrix{Float64}, y::Array{Float64,1},pos::Array{Float64,1} )
		(s,grad)=sigGrad(m,W1,y,pos);
		new(0,s,grad,Array{Tuple{Int,Int},1}(),W1,y)
	end
end
function tildeParameter(model::Model,ell::Int)
	@assert(model.L+1>=ell)
	@assert(1<=ell)
nPrev= (ell==1 ? model.n0 : model.n[ell-1])
nEll=(ell<=model.L ? model.n[ell] : 1)
n0=nEll*(nPrev+1)
pos=Array{Float64,1}(undef,n0)
for j = 1:nEll
for i = 1:nPrev
pos[(j-1)*(nPrev+1)+i]=model.W[ell][j,i]
end
pos[(j)*(nPrev+1)]=model.b[ell][j]
end
return pos
end
mutable struct TrainerState
	duplicateGroups::Array{Array{Int,1},1}
	rewrite::ModelRewrite
	Apseudo::Array{Float64,2}
	branches::Array{DataBranch,1}
	pos::Array{Float64,1} # position for theta_l
	critical::Array{Tuple{Int,Int,Int},1}
	gradient::Array{Float64,1}
	negModifiedGrad::Array{Float64,1}
	function TrainerState(model::Model,ell::Int,Xdata::Matrix, Ydata::Array{Float64,1})
		@assert(size(Xdata,2)==size(Ydata,1))
		@assert(size(Xdata,1)>0)
	modelRewrite=ModelRewrite(model,ell)
	rep=(ell<=model.L ? model.n[ell] : 1)
	outputs=[layerOutput(model,Xdata[:,i],ell-1) for i in 1:size(Xdata,2)]
	dup=duplicates(outputs)
	startmatrices=Array{Matrix,1}([constructTMatrix(rep,output) for output in outputs]);
	pos=tildeParameter(model,ell)
	branches=[DataBranch(modelRewrite,startmatrices[i],[Ydata[i]],pos) for i in 1:N];
	for k = 1:length(dup)
		representative=k
		for i = dup[k]
			branches[i].duplicateRepresentative=representative
		end
	end
	grad=branches[1].grad
	for i = 2:size(branches,1)
		grad+=branches[i].grad
	end
	critical=Array{Tuple{Int,Int,Int},1}()
	new(dup,modelRewrite,zeros(0,modelRewrite.n0),branches,pos,critical,grad,-grad)
	end
end
function sortfunction(x,y)
	if x[1]<y[1]
		return true;
	elseif x[1]==y[1]
		if x[2]<y[2]
			return true;
		end
	end
	return false
end
function advanceMax(rewrite::ModelRewrite,branch::DataBranch,pos::Array{Float64,1},v::Array{Float64,1},cap::Float64=-0.1)
s=branch.s
critical=deepcopy(branch.critical)
sort!(critical,lt=sortfunction)
criticalIdx=1;
considerCritical=(criticalIdx<=length(critical))
nextCriticalLayer=0;
if considerCritical
	nextCriticalLayer=first(critical[criticalIdx])	
end
change=Array{Tuple{Int,Int,Bool,Float64},1}()
sizehint!(change,10)
α=branch.W1*pos
β=branch.W1*v
t=Inf64
for l = 1:rewrite.L
# global considerCritical, α, β, t
if l==rewrite.L
	α-= branch.y
end
for j = 1:rewrite.n[l]
if considerCritical
if nextCriticalLayer==l
if critical[criticalIdx][2]==j
criticalIdx+=1
if criticalIdx>length(critical)
considerCritical=false
else
nextCriticalLayer=first(critical[criticalIdx])
end
continue;
end
end
end
if j>rewrite.nNoDuplicate[l]
continue;
end
if β[j]!=0
τ = -α[j]/β[j]
if (((s[l][j]==true)&&(β[j]<0))||((s[l][j]==false)&&(β[j]>0))) && τ>cap
if τ<=t+1e-10
if τ<t-1e-10
empty!(change)
t=τ
end
push!(change,(l,j,(β[j]>0),τ))
end
end
end
end
if l<rewrite.L
	α = rewrite.W2[l]*(s[l].*α)+rewrite.bNeedAdjustEnd[l+1]
	β = rewrite.W2[l]*(s[l].*β)
end
end
return (t,change)
end
function advanceMaxJoint(state::TrainerState, pos::Array{Float64,1},v::Array{Float64,1},cap::Float64=-0.1)
branches=state.branches
branches::Array{DataBranch,1}
amax=[advanceMax(state.rewrite,branches[i],pos,v,cap) for i in 1:size(branches,1)]
imin=0
ival=0.0
t=Inf64
change=Array{Tuple{Int,Int,Bool,Float64},1}()
for i = 1:length(amax)
	(tnew,changenew)=amax[i]
	if length(last(amax[i]))>0&&t>tnew
		t=tnew
		change=changenew
		imin=i
	end
end
return (imin,(t,change))
end
function branchLoss(rewrite::ModelRewrite,W1::Matrix,y::Array{Float64,1},pos::Array{Float64,1})
	#TODO: Variante mit s statt relu
tmp=W1*pos
for l = 1:rewrite.L-1
	tmp=ReLU.(tmp)
	tmp=rewrite.W2[l]*tmp+rewrite.bNeedAdjustEnd[l+1]
end
tmp-=y
if tmp[1]<0
	return -tmp
else
	return tmp
end
end
function loss(rewrite::ModelRewrite,branches::Array{DataBranch,1},pos::Array{Float64,1})
	return sum([branchLoss(rewrite,branches[i].W1,branches[i].y,pos) for i in 1:length(branches)])
end
function loss(state::TrainerState)
	return loss(state.rewrite,state.branches,state.pos)
end
function orientedNormalVec(rewrite::ModelRewrite,branch::DataBranch,neuronPos::Tuple{Int,Int})
	if neuronPos[1]==1
		if branch.s[1][neuronPos[2]]==true
			return branch.W1[neuronPos[2],:]
		else
			return -branch.W1[neuronPos[2],:]
		end
	else
		γ=rewrite.W2[neuronPos[1]-1][neuronPos[2],:]
		for l = (neuronPos[1]-1):-1:2
			γ=rewrite.W2[l-1]'*(γ.*branch.s[l])
		end
		γ=branch.W1'*(γ.*branch.s[1])
		if branch.s[neuronPos[1]][neuronPos[2]]==false
			return -γ
		else 
			return γ
		end
	end
end
function innerProductsOrientedNormalVectors(rewrite::ModelRewrite,branch::DataBranch,v::Array{Float64,1})
out=Array{Array{Float64,1},1}()
γ=(branch.W1)*v
for l = 1:rewrite.L
	r=Array{Float64,1}(undef,rewrite.n[l])
	sl=branch.s[l]
	for j = 1:rewrite.n[l]
	if sl[j]==true
	r[j]=γ[j]
	else
	r[j]=-γ[j]
	end
	end
	push!(out,r)
	if l<rewrite.L
		γ= rewrite.W2[l]*(sl.*γ)
	end
end
	return out
end
function PseudoInverseAddColIP(Apseudo::Matrix,AinnerProducts::Array{Float64,1},newCol::Array{Float64,1})
	# construct AnewPseudo
	AnewPseudo=zeros(size(Apseudo)[1]+1,size(Apseudo)[2])
	orthogonalPart=newCol-Apseudo'*AinnerProducts
	for i = 1:size(Apseudo,1)
		axis=Apseudo[i,:]
		adjustedAxis=axis-dot(axis,newCol)/dot(orthogonalPart,newCol)*orthogonalPart
		AnewPseudo[i,:]=adjustedAxis
	end
	AnewPseudo[size(AnewPseudo,1),:]=1/(dot(orthogonalPart,newCol))*orthogonalPart
	return AnewPseudo
end
function extractCriticalInnerProducts(criticalPositions::Array{Tuple{Int64,Int64,Int64},1},innerProducts::Array{Array{Array{Float64,1},1},1})
	out=Array{Float64,1}(undef,size(criticalPositions,1))
	for i = 1:length(criticalPositions)
		out[i]=innerProducts[criticalPositions[i][1]][criticalPositions[i][2]][criticalPositions[i][3]]
	end
	return out
end
function addCritical!(rewrite::ModelRewrite,state::TrainerState,pos::Tuple{Int,Int,Int})
	branch=state.branches[pos[1]]
	normalvec=orientedNormalVec(rewrite,branch,(pos[2],pos[3]))
	iproducts=[innerProductsOrientedNormalVectors(rewrite,branch,normalvec) for branch in state.branches]
	criticalInnerProducts = extractCriticalInnerProducts(state.critical,iproducts)
	state.Apseudo=PseudoInverseAddColIP(state.Apseudo,criticalInnerProducts,normalvec)
	push!(state.critical,pos)
	push!(branch.critical,(pos[2],pos[3]))
	return normalvec
end
function project(rewrite::ModelRewrite, state::TrainerState,vec::Array{Float64,1})
	iproducts=[innerProductsOrientedNormalVectors(rewrite,branch,vec) for branch in state.branches]
	coordinates = extractCriticalInnerProducts(state.critical,iproducts)
	return state.Apseudo' * coordinates
end
function step(state::TrainerState)
	v=state.negModifiedGrad
	if all(x->abs(x)<1e-10,v)
		return ((0,0,0),[],0)
	end
	rewrite=state.rewrite
	(i,(t,change))=advanceMaxJoint(state,state.pos,v)
	if i==0 #no step possible
		return ((-1,-1,-1),[],0)
	end
	state.pos=state.pos+t*v
	############################### add critical
	pos=(i,change[1][1],change[1][2])
	normalvec=addCritical!(rewrite,state,pos)
	if size(state.Apseudo,1)<rewrite.n0
		############################### orthogonalize
		v=state.negModifiedGrad
		state.negModifiedGrad=v-project(rewrite,state,v)
	else
		state.negModifiedGrad=zeros(rewrite.n0)
	end
	return (pos,normalvec,t)
end
function changeActivationPattern!(rewrite::ModelRewrite,branch::DataBranch,pos::Tuple{Int,Int},newVal::Bool)
	l=pos[1]
	j=pos[2]
	branch.s[l][j]=newVal
	if l>1
		diff=rewrite.n[l]-rewrite.nNoDuplicate[l]
		if diff>0
			lower=rewrite.nNoDuplicate[l]-diff+1
			upper=rewrite.nNoDuplicate[l]
			if j>=lower || j<=upper
				branch.s[l][j+diff]=!newVal
			elseif j>upper
				branch.s[l][j-diff]=!newVal
			end
		end
	end
end
function computeAPseudo(state::TrainerState)
A=Matrix{Float64}(undef,state.rewrite.n0,size(state.critical,1))
for i=1:size(state.critical,1)
	pos=state.critical[i]
	A[:,i]=orientedNormalVec(state.rewrite,state.branches[pos[1]],(pos[2],pos[3]))
end
return inv(A'A)A'
end
function projectSkip(state::TrainerState,v::Array{Float64,1},skipIndex::Int)
	iproducts=[innerProductsOrientedNormalVectors(state.rewrite,branch,v) for branch in state.branches]
	coordinates = extractCriticalInnerProducts(state.critical,iproducts)
	coordinates[skipIndex]=0.0;
	return state.Apseudo'*coordinates
end
function gradient(rewrite::ModelRewrite,branch::DataBranch)
if rewrite.L==1
if branch.s[1][1]==true
return branch.W1[1,:]
else
return -branch.W1[1,:]
end
else
γ=rewrite.W2[rewrite.L-1][1,:]
for l = (rewrite.L-1):-1:2
γ=rewrite.W2[l-1]'*(γ.*branch.s[l])
end
γ=branch.W1'*(γ.*branch.s[1])
if branch.s[rewrite.L][1]==false
return -γ
else 
return γ
end
end
end
function bestIndex(state::TrainerState, v::Array{Float64,1})
maxVal=0.0;
maxIndex=0;
for i = 1:size(state.Apseudo,1)
	newVal=dot(state.Apseudo[i,:],v)/sqrt(dot(state.Apseudo[i,:],state.Apseudo[i,:]))
	if newVal>maxVal
		maxIndex=i;
		maxVal=newVal
	end
end
return maxIndex
end
function PseudoInverseRemoveCol(Apseudo,colId::Int)
	axis=Apseudo[colId,:]
	AnewPseudo=zeros(size(Apseudo,1)-1,size(Apseudo,2))
	i2=1;
	for i1 = 1:(size(Apseudo,1)-1)
		if i2==colId
			i2+=1
		end
		currentAxis=Apseudo[i2,:]
		newAxis=currentAxis-dot(currentAxis,axis)/ dot(axis,axis)*axis
		AnewPseudo[i1,:]=newAxis
		i2+=1
	end
	return AnewPseudo
end
function removeCritical!(state::TrainerState, i::Int)
dataindex=state.critical[i][1]
ind=findfirst(x->x==(state.critical[i][2],state.critical[i][3]), state.branches[dataindex].critical)
deleteat!(state.branches[dataindex].critical, ind)
deleteat!(state.critical,i)
end
function duplicates(outputs::Array{Array{Float64,1},1})
D=Dict()
for i = 1:length(outputs)
	o=outputs[i]
if haskey(D,o)
	push!(D[o],i)
else
	D[o]=Set{Int}(i)
	end
end
a=Array{Array{Int,1},1}()
for p = D
	if length(last(p))>1
		push!(a,collect(last(p)))
		# println(p)
	end
end
return a
end
