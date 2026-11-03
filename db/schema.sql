-- Supabase SQL Editor에서 한 번 실행한다.
-- 프로젝트 설정: Data API 켜기, 새 테이블 자동 공개 끄기, 자동 RLS 켜기
--
-- 테이블 세 개 (2026-11-03 마이그레이션 split_sessions_suggestions_feedback, 그 전엔 events 하나)
--   sessions     브리지 실행 1회 = 1행. 버전은 여기에만 둔다
--   suggestions  제안 1건 = 1행
--   feedback     사용자 반응 1건 = 1행 (shown, accept)
-- 브리지가 logs/rows.jsonl에 남긴 뒤 10초마다 테이블별로 보낸다 (bridge.py의 TABLES와 칸이 같아야 한다).

create table public.sessions (
  session text primary key,
  started_at timestamptz not null,
  bridge_version text,   -- 브리지 실행 때 git describe --always --dirty
  fw_version text,       -- 펌웨어 빌드 때 ESP-IDF가 넣은 git describe (-dirty면 커밋 안 된 변경 포함)
  fw_build text,         -- 펌웨어 ELF sha256 앞 8자리 (-dirty 빌드끼리 구분)
  model_version text,    -- model.bin 끝에 export.py가 쓴 <체크포인트 이름>-<내용 sha256 앞 8자리>
  vocab_size int,
  port text
);

create table public.suggestions (
  session text not null,
  req int not null,
  ts timestamptz not null,
  prefix_tail text,          -- 커서 앞 코드의 끝 200자
  completion text,
  cached boolean not null default false,  -- 직전 요청과 앞부분이 같아 보드를 부르지 않음
  latency_ms int,            -- 브리지 기준 왕복 (USB 포함)
  gen_ms int,                -- 보드에서 잰 생성 시간 (캐시면 0)
  prompt_tokens int,         -- 앞부분의 원래 토큰 수
  dropped_tokens int,        -- 줄인 어휘에 없어 버린 토큰 수 (모델이 못 본 입력)
  gen_tokens int,            -- 생성한 토큰 수 (캐시면 0)
  -- 보드가 생성을 멈춘 이유
  --   eos 모델이 끝 토큰 선택 · newline 한 줄 완성 · max_tokens 32토큰 다 씀(반복 의심)
  --   heal_mismatch 힐링 문자열과 맞는 토큰 없음 · blank 공백뿐이라 버림 · seq_full 위치 256 다 참 · empty_input 입력 없음
  stop_reason text check (stop_reason in ('eos', 'newline', 'max_tokens', 'heal_mismatch', 'blank', 'seq_full', 'empty_input')),
  heap_min int,
  -- 브리지가 codeop로 검사한 '앞부분 + 완성'의 문법
  --   ok · incomplete(블록·괄호가 이어질 차례) · error · prefix_error(완성 전 줄들이 이미 오류) · null(앞부분이 1024자로 잘려 판단 불가)
  syntax text check (syntax in ('ok', 'incomplete', 'error', 'prefix_error')),
  syntax_error text,
  primary key (session, req)
);

create table public.feedback (
  session text not null,
  req int not null,
  kind text not null check (kind in ('shown', 'accept')),  -- shown: 확장이 VS Code에 제안을 넘김, accept: Tab으로 수락
  ts timestamptz not null,
  primary key (session, req, kind)
);

-- 공개 키(anon)에는 넣기만 허용.
-- ON CONFLICT(중복 무시)는 select 권한과 읽기 정책이 필요해서 쓰지 않는다.
-- 중복은 브리지가 409를 받으면 한 행씩 다시 넣으며 건너뛴다 (bridge.py supabase_post).
alter table public.sessions enable row level security;
alter table public.suggestions enable row level security;
alter table public.feedback enable row level security;
grant insert on public.sessions, public.suggestions, public.feedback to anon;
create policy "anon insert only" on public.sessions for insert to anon with check (true);
create policy "anon insert only" on public.suggestions for insert to anon with check (true);
create policy "anon insert only" on public.feedback for insert to anon with check (true);

-- 버전 조합별 성적표. security_invoker라 anon은 읽을 수 없다 (대시보드 SQL Editor에서 select * from scorecard)
create view public.scorecard with (security_invoker = true) as
select se.model_version, se.fw_version, se.bridge_version,
       count(*)                                                        as suggestions,
       count(*) filter (where s.cached)                                as cached,
       count(f_shown.req)                                              as shown,
       count(f_acc.req)                                                as accepted,
       round(100.0 * count(f_acc.req) / nullif(count(f_shown.req), 0), 1) as accept_pct_of_shown,
       round(100.0 * count(*) filter (where coalesce(s.completion, '') = '') / count(*), 1) as empty_pct,
       round(avg(s.latency_ms) filter (where not s.cached))            as avg_latency_ms,
       round(avg(s.gen_ms) filter (where not s.cached))                as avg_gen_ms,
       round(avg(s.prompt_tokens))                                     as avg_prompt_tokens,
       round(100.0 * sum(s.dropped_tokens) / nullif(sum(s.prompt_tokens), 0), 1) as dropped_token_pct,
       round(100.0 * count(*) filter (where s.syntax = 'error')
             / nullif(count(*) filter (where s.syntax in ('ok', 'incomplete', 'error')), 0), 1) as syntax_error_pct,
       round(100.0 * count(f_acc.req) filter (where s.syntax = 'error')
             / nullif(count(f_acc.req) filter (where s.syntax in ('ok', 'incomplete', 'error')), 0), 1) as accepted_syntax_error_pct,
       min(s.heap_min)                                                 as min_heap
from public.suggestions s
join public.sessions se using (session)
left join public.feedback f_shown on f_shown.session = s.session and f_shown.req = s.req and f_shown.kind = 'shown'
left join public.feedback f_acc   on f_acc.session = s.session and f_acc.req = s.req and f_acc.kind = 'accept'
group by se.model_version, se.fw_version, se.bridge_version;

-- 멈춘 이유 분포 (빈 완성·반복의 원인 구분)
create view public.stop_reasons with (security_invoker = true) as
select se.model_version, se.fw_version, s.stop_reason, count(*) as n,
       count(*) filter (where coalesce(s.completion, '') = '') as empty
from public.suggestions s join public.sessions se using (session)
where not s.cached
group by 1, 2, 3;

-- 예전 events 테이블(2026-11-03 13:50 이전 기록)은 지우지 않고 남겨 두었다. 그 데이터는 위 세 테이블로 옮겼다
-- (fw_version·bridge_version·prompt_tokens·stop_reason 등은 당시 기록하지 않아 비어 있다).
