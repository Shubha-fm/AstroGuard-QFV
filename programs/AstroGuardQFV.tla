------------------------------ MODULE AstroGuardQFV ------------------------------
EXTENDS Naturals, Sequences, TLC

CONSTANTS Objects, MaxVersion, MaxRetry

Stages == {"Received", "Inferencing", "Assessed", "Retry",
           "Released", "Review", "Withheld"}
Decisions == {"None", "Release", "Review", "Withhold"}

VARIABLES stage, modelVersion, assessedVersion,
          confidenceOK, uncertaintyOK, qualityOK, sourceOK,
          retryCount, decision

vars == <<stage, modelVersion, assessedVersion,
          confidenceOK, uncertaintyOK, qualityOK, sourceOK,
          retryCount, decision>>

TypeOK ==
  /\ stage \in [Objects -> Stages]
  /\ modelVersion \in 1..MaxVersion
  /\ assessedVersion \in [Objects -> 0..MaxVersion]
  /\ confidenceOK \in [Objects -> BOOLEAN]
  /\ uncertaintyOK \in [Objects -> BOOLEAN]
  /\ qualityOK \in [Objects -> BOOLEAN]
  /\ sourceOK \in [Objects -> BOOLEAN]
  /\ retryCount \in [Objects -> 0..MaxRetry]
  /\ decision \in [Objects -> Decisions]

Init ==
  /\ stage = [o \in Objects |-> "Received"]
  /\ modelVersion = 1
  /\ assessedVersion = [o \in Objects |-> 0]
  /\ confidenceOK = [o \in Objects |-> FALSE]
  /\ uncertaintyOK = [o \in Objects |-> FALSE]
  /\ qualityOK = [o \in Objects |-> FALSE]
  /\ sourceOK = [o \in Objects |-> FALSE]
  /\ retryCount = [o \in Objects |-> 0]
  /\ decision = [o \in Objects |-> "None"]

Infer(o) ==
  /\ stage[o] \in {"Received", "Retry"}
  /\ stage' = [stage EXCEPT ![o] = "Inferencing"]
  /\ decision' = [decision EXCEPT ![o] = "None"]
  /\ UNCHANGED <<modelVersion, assessedVersion,
                 confidenceOK, uncertaintyOK, qualityOK, sourceOK, retryCount>>

Assess(o) ==
  /\ stage[o] = "Inferencing"
  /\ \E c, u, q, s \in BOOLEAN:
       /\ stage' = [stage EXCEPT ![o] = "Assessed"]
       /\ assessedVersion' = [assessedVersion EXCEPT ![o] = modelVersion]
       /\ confidenceOK' = [confidenceOK EXCEPT ![o] = c]
       /\ uncertaintyOK' = [uncertaintyOK EXCEPT ![o] = u]
       /\ qualityOK' = [qualityOK EXCEPT ![o] = q]
       /\ sourceOK' = [sourceOK EXCEPT ![o] = s]
       /\ UNCHANGED <<modelVersion, retryCount, decision>>

Retry(o) ==
  /\ stage[o] \in {"Inferencing", "Assessed"}
  /\ retryCount[o] < MaxRetry
  /\ stage' = [stage EXCEPT ![o] = "Retry"]
  /\ retryCount' = [retryCount EXCEPT ![o] = @ + 1]
  /\ assessedVersion' = [assessedVersion EXCEPT ![o] = 0]
  /\ decision' = [decision EXCEPT ![o] = "None"]
  /\ UNCHANGED <<modelVersion,
                 confidenceOK, uncertaintyOK, qualityOK, sourceOK>>

ReloadModel ==
  /\ modelVersion < MaxVersion
  /\ modelVersion' = modelVersion + 1
  /\ UNCHANGED <<stage, assessedVersion,
                 confidenceOK, uncertaintyOK, qualityOK, sourceOK,
                 retryCount, decision>>

Release(o) ==
  /\ stage[o] = "Assessed"
  /\ assessedVersion[o] = modelVersion
  /\ confidenceOK[o] /\ uncertaintyOK[o] /\ qualityOK[o] /\ sourceOK[o]
  /\ stage' = [stage EXCEPT ![o] = "Released"]
  /\ decision' = [decision EXCEPT ![o] = "Release"]
  /\ UNCHANGED <<modelVersion, assessedVersion,
                 confidenceOK, uncertaintyOK, qualityOK, sourceOK, retryCount>>

Review(o) ==
  /\ stage[o] = "Assessed"
  /\ ~(confidenceOK[o] /\ uncertaintyOK[o] /\ qualityOK[o] /\ sourceOK[o])
  /\ stage' = [stage EXCEPT ![o] = "Review"]
  /\ decision' = [decision EXCEPT ![o] = "Review"]
  /\ UNCHANGED <<modelVersion, assessedVersion,
                 confidenceOK, uncertaintyOK, qualityOK, sourceOK, retryCount>>

Next ==
  \/ \E o \in Objects: Infer(o)
  \/ \E o \in Objects: Assess(o)
  \/ \E o \in Objects: Retry(o)
  \/ ReloadModel
  \/ \E o \in Objects: Release(o)
  \/ \E o \in Objects: Review(o)

Spec == Init /\ [][Next]_vars

NoUnsafeRelease ==
  \A o \in Objects:
    decision[o] = "Release" =>
      (confidenceOK[o] /\ uncertaintyOK[o] /\ qualityOK[o] /\ sourceOK[o])

NoStaleRelease ==
  \A o \in Objects:
    decision[o] = "Release" => assessedVersion[o] = modelVersion

RetryInvalidatesAssessment ==
  \A o \in Objects:
    stage[o] = "Retry" => decision[o] # "Release"

=============================================================================
