include("../models/random.jl")
include("util/helpers.jl")

function branchReLUArguments(rewrite::ModelRewrite,branch::DataBranch,pos::Array{Float64,1})
A=Array{Array{Float64,1},1}()
tmp=branch.W1*pos
for l = 1:rewrite.L-1
	tmp+=rewrite.bNeedAdjustEnd[l]
	push!(A,tmp)
	tmp=branch.s[l].*tmp
	tmp=rewrite.W2[l]*tmp
end
tmp+=rewrite.bNeedAdjustEnd[rewrite.L]-branch.y
push!(A,tmp)
return A
end
function criticalCoordinates(state::TrainerState)
	return Array{Float64,1}([branchReLUArguments(state.rewrite,state.branches[c[1]],state.pos)[c[2]][c[3]] for c in state.critical])
end
function coordinateDiff(state::TrainerState)
	return maximum(abs.( criticalCoordinates(state)))
end
function updateTrained!(model::Model,ell::Int,pos::Array{Float64,1})
	wsize=size(model.W[ell])
	for i = 1:wsize[1]
		model.W[ell][i,:]=pos[((i-1)*(wsize[2]+1)+1):((i-1)*(wsize[2]+1)+wsize[2])]
		model.b[ell][i]=pos[i*(wsize[2]+1)]
	end
end
function changeActivationPattern!(state::TrainerState,pos::Tuple{Int,Int,Int},newActivation::Bool)
	# changeActivationPattern!(state.rewrite,state.branches[pos[1]],(pos[2],pos[3]),newActivation)
	# return
	if state.branches[pos[1]].duplicateRepresentative>0 && pos[2]<state.rewrite.L
		representative=Int(state.branches[pos[1]].duplicateRepresentative)
		println("multiple___________________________________________change")
		for branch=state.branches
			if branch.duplicateRepresentative==representative
				changeActivationPattern!(state.rewrite,branch,(pos[2],pos[3]),newActivation)
			end
		end
	else
		changeActivationPattern!(state.rewrite,state.branches[pos[1]],(pos[2],pos[3]),newActivation)
	end
end
function updateGradient!(state::TrainerState,i::Int)
	if state.branches[i].duplicateRepresentative>0 
		representative=Int(state.branches[i].duplicateRepresentative)
		println("multiple gradients_____________________________________")
		for branch=state.branches
			if branch.duplicateRepresentative==representative
				branch.grad=gradient(state.rewrite,branch)
			end
		end
	else
		state.branches[i].grad=gradient(state.rewrite,state.branches[i])
	end
	state.gradient=sum([branch.grad for branch in state.branches])
	return state.gradient
end
function sigdiffs(state::TrainerState)
diffs=Array{Tuple{Int,Int,Int},1}()
assumed=[branch.s for branch in state.branches]
real=[sigGrad(state.rewrite,branch.W1,branch.y,state.pos)[1] for branch in state.branches]
for i = 1:length(real)
r=real[i]
for l = 1:length(r)
vals=r[l]
for j = 1:length(vals)
if vals[j]!=assumed[i][l][j]
push!(diffs,(i,l,j))
end
end
end
end
return diffs
end
function argDiffs(state::TrainerState)
diffs=sigdiffs(state)
args=[branchReLUArguments(state.rewrite,branch,state.pos) for branch in state.branches]
return [args[d[1]][d[2]][d[3]] for d in diffs]
end



##############################



##############################




randmodel=randomModel(4,[5,5,5,5,2])
model=randmodel
N=1000
# input parameters
Ydata=rand(Uniform(-3,3),N)
Xdata=rand(Uniform(-3,3),randmodel.n0,N)

ell=1
state=TrainerState(randmodel,ell,Xdata,Ydata)
state.gradient
(code,state,pos)=trainL1(model,ell,Xdata,Ydata);

updateTrained!(model,ell,pos)

#######################
#
#
#


grads=[branch.grad for branch in state.branches]

sum(grads)

grads[34]

state

sigdiffs(state)
adiffs=argDiffs(state)
state.duplicateGroups


state=TrainerState(randmodel,ell,Xdata,Ydata)

i=1
[state.branches[k].s for k in state.duplicateGroups[i]]

[state.branches[k].W1 for k in state.duplicateGroups[i]]
[state.branches[k].y for k in state.duplicateGroups[i]]

[branchReLUArguments(state.rewrite,state.branches[k],state.pos) for k in state.duplicateGroups[i]]

state.branches

state.branches[180].s
state.branches[106].s

state.duplicateGroups


index=1

pos=state.critical[index]
newActivation=!state.branches[pos[1]].s[pos[2]][pos[3]]
changeActivationPattern!(state,pos,newActivation)

updateGradient!(state,pos[1])
state.Apseudo=computeAPseudo(state) # TODO: different result, check
println("diff before: $(coordinateDiff(state))")
i=bestIndex(state,-state.gradient)

state.negModifiedGrad=state.Apseudo[i,:]
state.Apseudo= PseudoInverseRemoveCol(state.Apseudo,i)
removeCritical!(state,i)
(pos,normalvec,t)=step(state)
l=loss(state)









trace=[]
state=TrainerState(model,ell,Xdata,Ydata)
##############################
# findVertex
##############################
println( loss(state))
while size(state.critical,1) < state.rewrite.n0
	# global trace
	(pos,normalvec,t)=step(state)
	if pos[1]<=0
		#if pos[1]==0 then gradient is zero, 
		#if pos[1]==-1 then function can be made arbitrarily small
		if pos[1]==0
			println("gradient is zero!")
		end
		if pos[1]==-1
			println("can be made arbitrary small!")
		end
		return (pos[1],trace,state.pos)
	end
	state.Apseudo=computeAPseudo(state)
	l=loss(state)
	# println("diff: $(coordinateDiff(state))")
	push!(trace,(deepcopy(state),deepcopy(normalvec), l))
	println(l,det(state.Apseudo*state.Apseudo'))
end;

index=1
while true
	global state, index
pos=state.critical[index]
newActivation=!state.branches[pos[1]].s[pos[2]][pos[3]]
changeActivationPattern!(state,pos,newActivation)
normalvec=orientedNormalVec(state.rewrite,state.branches[pos[1]],(pos[2],pos[3]))
projected=projectSkip(state, normalvec,index)
orthogonal=normalvec-projected
axis=(1/dot(orthogonal,normalvec))* orthogonal
state.Apseudo[index,:]=axis
updateGradient!(state,pos[1])
state.Apseudo=computeAPseudo(state) # TODO: different result, check
println("diff before: $(coordinateDiff(state))")
push!(trace,(deepcopy(state),deepcopy(normalvec), loss(state)))
state=deepcopy(trace[end][1])
i=bestIndex(state,-state.gradient)
if i>0
	state.negModifiedGrad=state.Apseudo[i,:]
	state.Apseudo= PseudoInverseRemoveCol(state.Apseudo,i)
	removeCritical!(state,i)
	(pos,normalvec,t)=step(state)
	l=loss(state)
	if t>0 && l[1]>last(last(trace))[1]
		println("Increases in value!")
		println(l)
		return (-2,trace,state.pos) #-2 = increases in function value
	end
	if pos[1]<=0 
		if pos[1]==0
			println("gradient is zero!")
		end
		if pos[1]==-1
			println("can be made arbitrary small!")
		end
		return (pos[1],trace,state.pos) #return information code "pos[1]"
	end
	index=1
	println(loss(state)[1])
else
	index+=1
	if index>state.rewrite.n0
		println("finished")
		return(1,trace,state.pos)
		break;
	end
end
end

#######################

function  trainL1(model::Model,ell::Int,Xdata::Matrix,Ydata::Array{Float64,1})
trace=[]
prevloss=Inf
state=TrainerState(model,ell,Xdata,Ydata)
##############################
# findVertex
##############################
println( loss(state))
while size(state.critical,1) < state.rewrite.n0
	# global trace
	(pos,normalvec,t)=step(state)
	if pos[1]<=0
		#if pos[1]==0 then gradient is zero, 
		#if pos[1]==-1 then function can be made arbitrarily small
		if pos[1]==0
			println("gradient is zero!")
		end
		if pos[1]==-1
			println("can be made arbitrary small!")
		end
		return (pos[1],state,state.pos)
	end
	state.Apseudo=computeAPseudo(state)
	l=loss(state)
	prevloss=l
	# push!(trace,(deepcopy(state),deepcopy(normalvec), l))
	println(l,det(state.Apseudo*state.Apseudo'))
end
##############################
# change the region
##############################
index=1
while true
	# global state, index
pos=state.critical[index]
#
newActivation=!state.branches[pos[1]].s[pos[2]][pos[3]]
changeActivationPattern!(state,pos,newActivation)
#
# compute new axis
normalvec=orientedNormalVec(state.rewrite,state.branches[pos[1]],(pos[2],pos[3]))
#
# normalvec=-normalvec #TODO: check!
projected=projectSkip(state, normalvec,index)
orthogonal=normalvec-projected
axis=(1/dot(orthogonal,normalvec))* orthogonal
state.Apseudo[index,:]=axis
#
# compute new gradient
updateGradient!(state,pos[1])
##############################
# select best axis to continue
##############################
state.Apseudo=computeAPseudo(state) # TODO: different result, check
cdiff=coordinateDiff(state)
if cdiff>1e-8
	println("coordinate diff: $(cdiff)")
end
adiffs=argDiffs(state)
if length(adiffs)>0
	adiff=maximum(abs.(adiffs))
	if adiff>1e-8
		println("argdiff: $(adiff)")
	end
end
# push!(trace,(deepcopy(state),deepcopy(normalvec), loss(state)))
# state=deepcopy(trace[end][1])
i=bestIndex(state,-state.gradient)
if i>0
	state.negModifiedGrad=state.Apseudo[i,:]
	state.Apseudo= PseudoInverseRemoveCol(state.Apseudo,i)
	removeCritical!(state,i)
	(pos,normalvec,t)=step(state)
	l=loss(state)
	if t>0 && l[1]>prevloss[1]
		println("Increases in value!")
		println(l)
		return (-2,state,state.pos) #-2 = increases in function value
	end
	prevloss=l
	if pos[1]<=0 
		if pos[1]==0
			println("gradient is zero!")
		end
		if pos[1]==-1
			println("can be made arbitrary small!")
		end
		#if pos[1]==0 then gradient is zero, 
		#if pos[1]==-1 then function can be made arbitrarily small
		return (pos[1],state,state.pos) #return information code "pos[1]"
	end
	index=1
	println(loss(state)[1])
else
	index+=1
	if index>state.rewrite.n0
		println("finished")
		return(1,state,state.pos)
		break;
	end
end
end
end





