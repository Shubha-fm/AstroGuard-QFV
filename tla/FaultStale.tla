------------------------------ MODULE FaultStale ------------------------------
EXTENDS Naturals, Sequences, TLC

CONSTANTS Objects, MaxVersion, MaxRetry
Stages == {"Received", "Inferencing", "Assessed", "Retry", "Released", "Review", "Withheld"}
Decisions == {"None", "Release", "Review", "Withhold"}
VARIABLES stage, modelVersion, assessedVersion,
          confidenceOK, uncertaintyOK, qualityOK, sourceOK,
          retryCount, decision, actor
vars == <<stage, modelVersion, assessedVersion,
          confidenceOK, uncertaintyOK, qualityOK, sourceOK,
          retryCount, decision, actor>>
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
  /\ actor \in (Objects \cup {"System", "None"})

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
  /\ actor = "None"

Infer(o) ==
  /\ stage[o] \in {"Received", "Retry"}
  /\ stage' = [stage EXCEPT ![o] = "Inferencing"]
  /\ decision' = [decision EXCEPT ![o] = "None"]
  /\ actor' = o
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
       /\ actor' = o
       /\ UNCHANGED <<modelVersion, retryCount, decision>>

Retry(o) ==
  /\ stage[o] \in {"Inferencing", "Assessed"}
  /\ retryCount[o] < MaxRetry
  /\ stage' = [stage EXCEPT ![o] = "Retry"]
  /\ retryCount' = [retryCount EXCEPT ![o] = @ + 1]
  /\ assessedVersion' = [assessedVersion EXCEPT ![o] = 0]
  /\ decision' = [decision EXCEPT ![o] = "None"]
  /\ actor' = o
  /\ UNCHANGED <<modelVersion, confidenceOK, uncertaintyOK, qualityOK, sourceOK>>

ReloadModel ==
  /\ modelVersion < MaxVersion
  /\ modelVersion' = modelVersion + 1
  /\ actor' = "System"
  /\ UNCHANGED <<stage, assessedVersion,
                 confidenceOK, uncertaintyOK, qualityOK, sourceOK, retryCount, decision>>

Release(o) ==
  /\ stage[o] = "Assessed"
  /\ confidenceOK[o]
  /\ uncertaintyOK[o]
  /\ qualityOK[o]
  /\ sourceOK[o]
  /\ stage' = [stage EXCEPT ![o] = "Released"]
  /\ decision' = [decision EXCEPT ![o] = "Release"]
  /\ actor' = o
  /\ UNCHANGED <<modelVersion, assessedVersion,
                 confidenceOK, uncertaintyOK, qualityOK, sourceOK, retryCount>>

Review(o) ==
  /\ stage[o] = "Assessed"
  /\ sourceOK[o]
  /\ ~(confidenceOK[o] /\ uncertaintyOK[o] /\ qualityOK[o])
  /\ stage' = [stage EXCEPT ![o] = "Review"]
  /\ decision' = [decision EXCEPT ![o] = "Review"]
  /\ actor' = o
  /\ UNCHANGED <<modelVersion, assessedVersion,
                 confidenceOK, uncertaintyOK, qualityOK, sourceOK, retryCount>>

Withhold(o) ==
  /\ stage[o] = "Assessed"
  /\ ~sourceOK[o]
  /\ stage' = [stage EXCEPT ![o] = "Withheld"]
  /\ decision' = [decision EXCEPT ![o] = "Withhold"]
  /\ actor' = o
  /\ UNCHANGED <<modelVersion, assessedVersion,
                 confidenceOK, uncertaintyOK, qualityOK, sourceOK, retryCount>>

ObjectLocalAction(o) ==
  Infer(o) \/ Assess(o) \/ Retry(o) \/ Release(o) \/ Review(o) \/ Withhold(o)
Next ==
  \/ \E o \in Objects: ObjectLocalAction(o)
  \/ ReloadModel
Spec == Init /\ [][Next]_vars

NoUnsafeRelease ==
  \A o \in Objects:
    decision[o] = "Release" =>
       (confidenceOK[o] /\ uncertaintyOK[o] /\ qualityOK[o] /\ sourceOK[o])
FreshReleaseAction ==
  \A o \in Objects:
    ((decision'[o] = "Release") /\ (decision[o] # "Release")) =>
      assessedVersion[o] = modelVersion
FreshRelease == [][FreshReleaseAction]_vars
RetryInvalidatesAssessmentAction ==
  \A o \in Objects:
    ((stage'[o] = "Retry") /\ (stage[o] # "Retry")) =>
      ((assessedVersion'[o] = 0) /\ (decision'[o] = "None"))
RetryInvalidatesAssessment == [][RetryInvalidatesAssessmentAction]_vars
Local(o) ==
  <<stage[o], assessedVersion[o], confidenceOK[o], uncertaintyOK[o],
    qualityOK[o], sourceOK[o], retryCount[o], decision[o]>>
\* DO NOT write 'ObjectLocalAction(o) => UNCHANGED Local(other)'.
\* That implication is vacuously true on an illegally non-local action.
ObjectIsolationAction ==
  \A o \in Objects:
    (actor' = o) =>
      \A other \in Objects: (other # o) => UNCHANGED Local(other)
ObjectIsolation == [][ObjectIsolationAction]_vars
=============================================================================
