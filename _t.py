import sys; sys.stdout.reconfigure(encoding="utf-8")
from ragdoc.artifacts import CheckResult as R, CheckStatus as S
from ragdoc.attribution.verdict import Verdict, StageReport as SR
from ragdoc.config import RunConfig
from ragdoc.report import render_report
c=RunConfig(top_k=5)
def v(stage,reason,code,fix,sts,withheld=False):
    return Verdict(stage,code,reason,fix,tuple(SR(n,s,r) for n,s,r in sts),withheld)
ev={"position":23,"total":400,"top_k":5}
print(render_report(v("retrieval","ranked below","rank_below_top_k","raise top_k",[("extraction",S.PASS,R.passed()),("chunking",S.PASS,R.passed()),("retrieval",S.FAIL,R.failed("ranked below",ev)),("generation",S.NOT_REACHED,None)]),c));print()
print(render_report(v("extraction","lost","extraction_lost","try loader",[("extraction",S.FAIL,R.failed("lost",{})),("chunking",S.NOT_REACHED,R.not_reached()),("retrieval",S.NOT_REACHED,R.not_reached()),("generation",S.NOT_REACHED,None)]),c));print()
print(render_report(v(None,None,None,None,[("extraction",S.UNEVALUATED,R.unevaluated("x")),("chunking",S.NOT_REACHED,None),("retrieval",S.NOT_REACHED,None),("generation",S.NOT_REACHED,None)],True),c))
