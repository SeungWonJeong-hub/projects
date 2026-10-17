"""GEE 인증 — stdin 없이 2단계, 최신 notebook Flow(code.earthengine.google.com, OOB 아님).
①  python ee_auth.py url    → 인증 URL 출력 (승인→코드 복사)
②  python ee_auth.py CODE   → 코드로 토큰 교환·저장
"""
import sys
import ee.oauth as o
VF="/tmp/ee_verifier.txt"

if len(sys.argv)<2 or sys.argv[1]=="url":
    flow=o.Flow(auth_mode="notebook")
    open(VF,"w").write(flow.code_verifier)
    print("\n=== 이 URL을 브라우저에서 열고 승인 후, 인증코드를 채팅에 붙여주세요 ===\n")
    print(flow.auth_url)
else:
    code=sys.argv[1].strip()
    verifier=open(VF).read().strip()
    o._obtain_and_write_token(code, verifier, o.SCOPES)
    print("인증 저장 완료 →", o.get_credentials_path())
