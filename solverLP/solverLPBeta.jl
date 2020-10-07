
function branchReLUArguments(rewrite::ModelRewrite,branch::DataBranch,pos::Array{Float64,1})
A=Array{Array{Float64,1},1}()
tmp=branch.W1*pos
for l = 1:rewrite.L-1
	tmp+=rewrite.c[l]
	push!(A,tmp)
	tmp=branch.s[l].*tmp
	tmp=rewrite.W2[l]*tmp
end
tmp+=rewrite.c[rewrite.L]-branch.y
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
	if state.branches[pos[1]].duplicateGroup>0 && pos[2]<state.rewrite.L # the last layer can have different y values and should not be changed jointly
		representative=Int(state.branches[pos[1]].duplicateGroup)
		println("multiple___________________________________________change")
		for branch=state.branches
			if branch.duplicateGroup==representative
				changeActivationPattern!(state.rewrite,branch,(pos[2],pos[3]),newActivation)
			end
		end
	else
		changeActivationPattern!(state.rewrite,state.branches[pos[1]],(pos[2],pos[3]),newActivation)
	end
end
function updateGradient!(state::TrainerState,i::Int)
	if state.branches[i].duplicateGroup>0 
		representative=Int(state.branches[i].duplicateGroup)
		println("multiple gradients_____________________________________")
		for branch=state.branches
			if branch.duplicateGroup==representative
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

function trainL1(model::Model,ell::Int,Xdata::Matrix,Ydata::Array{Float64,1})
trace=[]
prevloss=Inf
global state=TrainerState(model,ell,Xdata,Ydata)
##############################
# findVertex
##############################
println( loss(state))
while size(state.critical,1) < state.rewrite.n0
	# global trace
	global prevState=deepcopy(state)
	(pos,normalvec,t)=step(state)
	if pos[1]<=0
		#if pos[1]==0 then gradient is zero, 
		#if pos[1]==-1 then function can be made arbitrarily small
		if pos[1]==0
			throw("gradient is zero")
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
	@warn("coordinate diff: $(cdiff)")
end
adiffs=argDiffs(state)
if length(adiffs)>0
	adiff=maximum(abs.(adiffs))
	if adiff>1e-8
		@warn "argdiff: $(adiff)"
	end
end
# push!(trace,(deepcopy(state),deepcopy(normalvec), loss(state)))
# state=deepcopy(trace[end][1])
i=bestIndex(state,-state.gradient)
if i>0
	global prevState=deepcopy(state)
	state.direction=state.Apseudo[i,:]
	state.Apseudo= PseudoInverseRemoveCol(state.Apseudo,i)
	removeCritical!(state,i)
	(pos,normalvec,t)=step(state)
	l=loss(state)
	if t>0 && l[1]>prevloss[1]
		@warn("Increase in loss value")
		println(l)
	end
	prevloss=l
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
