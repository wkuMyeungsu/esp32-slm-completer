"""제안의 문법 검사. 브리지(실사용 기록)와 train/eval_bin.py(평가·CI)가 같은 규칙을 쓴다. 의존성 없음."""
import codeop
import warnings

PREFIX_MAX = 1024  # 확장이 보내는 앞부분 최대 길이 (extension.ts의 slice(-1024))


def _compile(src: str) -> tuple[str, str | None]:
    """Python 파서(codeop, 대화형 셸이 '...'를 띄울지 정하는 모듈)로 검사. 덜 쓴 코드는 incomplete."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 잘못된 이스케이프 같은 SyntaxWarning은 오류가 아니다
        try:
            return ("ok" if codeop.compile_command(src, "<prefix>", "exec") else "incomplete"), None
        except (SyntaxError, ValueError, OverflowError) as e:
            return "error", f"{getattr(e, 'msg', e)} (line {getattr(e, 'lineno', '?')})"


def syntax_check(prefix: str, completion: str) -> tuple[str | None, str | None]:
    """앞부분 + 완성의 문법. 판단할 수 없으면 (None, None): 완성이 비었거나(모델이 낸 것이 없음)
    앞부분이 잘렸을 때(파일 앞쪽이 없음).
    오류가 앞부분만 넣었을 때와 같으면(메시지·줄이 같음) 원래 있던 오류라 모델 탓이 아니므로 prefix_error."""
    if not completion or len(prefix) >= PREFIX_MAX:
        return None, None
    status, msg = _compile(prefix + completion)
    if status == "error" and _compile(prefix) == (status, msg):
        return "prefix_error", msg
    return status, msg
