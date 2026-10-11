-- Builds one row per (player, salary season) for position players.
-- Every performance feature comes from the season BEFORE the salary year,
-- so the model only uses information a team would have had when it set the salary.

WITH batting_season AS (
    -- Collapse mid-season trades (multiple stints) into one row per player-season
    SELECT
        playerID,
        yearID,
        SUM(G)   AS G,
        SUM(AB)  AS AB,
        SUM(H)   AS H,
        SUM("2B") AS doubles,
        SUM("3B") AS triples,
        SUM(HR)  AS HR,
        SUM(RBI) AS RBI,
        SUM(SB)  AS SB,
        SUM(BB)  AS BB,
        SUM(SO)  AS SO,
        SUM(COALESCE(HBP, 0)) AS HBP,
        SUM(COALESCE(SF, 0))  AS SF
    FROM batting
    GROUP BY playerID, yearID
),

pitcher_flag AS (
    -- A player is treated as a pitcher in a season if most of his games were on the mound
    SELECT playerID, yearID,
           CASE WHEN SUM(G_p) * 2 > SUM(G_all) THEN 1 ELSE 0 END AS is_pitcher
    FROM appearances
    GROUP BY playerID, yearID
),

batting_with_history AS (
    SELECT
        b.*,
        -- Rate stats
        CASE WHEN b.AB > 0 THEN 1.0 * b.H / b.AB END AS AVG,
        CASE WHEN (b.AB + b.BB + b.HBP + b.SF) > 0
             THEN 1.0 * (b.H + b.BB + b.HBP) / (b.AB + b.BB + b.HBP + b.SF) END AS OBP,
        CASE WHEN b.AB > 0
             THEN 1.0 * (b.H + b.doubles + 2 * b.triples + 3 * b.HR) / b.AB END AS SLG,
        -- Window functions: experience and career totals up to and including this season
        ROW_NUMBER() OVER w_career                  AS seasons_played,
        SUM(b.G)  OVER w_career                     AS career_G,
        SUM(b.HR) OVER w_career                     AS career_HR,
        SUM(b.H)  OVER w_career                     AS career_H,
        -- Change in home runs vs. the prior season (momentum)
        b.HR - LAG(b.HR) OVER (PARTITION BY b.playerID ORDER BY b.yearID) AS HR_change
    FROM batting_season b
    WINDOW w_career AS (PARTITION BY b.playerID ORDER BY b.yearID
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
),

salary_season AS (
    -- Players traded mid-year can have two salary rows; keep the total
    SELECT playerID, yearID, SUM(salary) AS salary
    FROM salaries
    GROUP BY playerID, yearID
),

salary_context AS (
    -- Normalize pay against that year's league: salaries roughly 6x'd from 1985 to 2016
    SELECT
        s.*,
        PERCENT_RANK() OVER (PARTITION BY s.yearID ORDER BY s.salary) AS salary_pct_rank,
        AVG(s.salary)  OVER (PARTITION BY s.yearID)                   AS league_avg_salary
    FROM salary_season s
)

SELECT
    sc.playerID,
    p.nameFirst || ' ' || p.nameLast            AS player_name,
    sc.yearID                                   AS salary_year,
    sc.salary,
    sc.salary_pct_rank,
    sc.league_avg_salary,
    sc.salary / sc.league_avg_salary            AS salary_vs_league,
    sc.yearID - p.birthYear                     AS age,
    -- Prior-season performance
    h.G, h.AB, h.H, h.HR, h.RBI, h.SB, h.BB, h.SO,
    h.AVG, h.OBP, h.SLG, h.OBP + h.SLG          AS OPS,
    h.HR_change,
    -- Experience through the prior season
    h.seasons_played, h.career_G, h.career_HR, h.career_H
FROM salary_context sc
JOIN batting_with_history h
  ON h.playerID = sc.playerID
 AND h.yearID   = sc.yearID - 1
JOIN pitcher_flag pf
  ON pf.playerID = h.playerID
 AND pf.yearID   = h.yearID
JOIN people p
  ON p.playerID = sc.playerID
WHERE pf.is_pitcher = 0
  AND h.AB >= 100          -- enough plate appearances for rate stats to mean something
ORDER BY sc.yearID, sc.salary DESC;
