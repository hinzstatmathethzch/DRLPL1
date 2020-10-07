include("../models/random.jl")
include("util/helpers.jl")
include("./solverLPBeta.jl")



##############################



##############################




randmodel=randomModel(4,[4,2])
model=randmodel
N=1000
# input parameters
Ydata=rand(Uniform(-3,3),N)
Xdata=rand(Uniform(-3,3),randmodel.n0,N)

ell=2
state=TrainerState(randmodel,ell,Xdata,Ydata)
state.gradient
(code,state,pos)=trainL1(model,ell,Xdata,Ydata);




A=Matrix{Float64}(undef,state.rewrite.n0,size(state.critical,1))
for i=1:size(state.critical,1)
	pos=state.critical[i]
	A[:,i]=orientedNormalVec(state.rewrite,state.branches[pos[1]],(pos[2],pos[3]))
end

[p[1] in  state.duplicateGroups[1] for p in state.critical]
A

B=A[:,[5,6,7,8,10]]
det(B'B)

state.critical

b=state.branches[1]

b

cbs=[state.branches[p[1]] for p in state.critical]


updateTrained!(model,ell,pos)

trace=[]
prevloss=Inf
state=TrainerState(model,ell,Xdata,Ydata)
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
		return (pos[1],trace,state.pos)
	end
	state.Apseudo=computeAPseudo(state)
	l=loss(state)
	prevloss=l
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
		return(1,trace,state.pos)
		break;
	end
end
end

#######################






