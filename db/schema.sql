-- Supabase SQL Editor에서 한 번 실행한다.
-- 프로젝트 설정: Data API 켜기, 새 테이블 자동 공개 끄기, 자동 RLS 켜기
create table public.events (
  session text not null,
  req int not null,
  event text not null check (event in ('suggest', 'accept')),
  ts timestamptz not null,
  model_version text,
  prefix_tail text,
  completion text,
  latency_ms int,
  gen_ms int,
  tokens int,
  heap_min int,
  primary key (session, req, event)
);

alter table public.events enable row level security;

-- 공개 키(anon)에는 넣기만 허용
grant insert on public.events to anon;
create policy "anon insert only" on public.events
  for insert to anon with check (true);

-- 모델 버전별 실사용 성적표 (조회는 대시보드 SQL Editor에서)
-- select s.model_version,
--        count(*)                                   as suggestions,
--        round(100.0 * count(a.req) / count(*), 1)  as accept_pct,
--        round(avg(s.latency_ms))                   as avg_latency_ms,
--        min(s.heap_min)                            as min_heap
-- from events s
-- left join events a on a.session = s.session and a.req = s.req and a.event = 'accept'
-- where s.event = 'suggest'
-- group by s.model_version
-- order by s.model_version;
