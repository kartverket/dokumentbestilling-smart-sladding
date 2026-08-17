-- Statistikk per rettsstiftelse, kun for perioden ETTER 2026-07-24 14:04:00
WITH rb AS (
    SELECT DISTINCT ON (btd.dokument_aar, btd.dokument_nr, btd.embete)
        b.dato,
        btd.dokument_aar,
        btd.dokument_nr,
        btd.embete,
        bta.fil_revisjon_id,
        bta.uthentet_versjon,
        bd.rettsstiftelsestyper
    FROM bestilling b
             JOIN bestilling_tinglyst_dokument btd ON btd.bestilling_id = b.id
             JOIN bestilling_tinglyst_dokument_apnet bta ON bta.bestilling_id = b.id
             JOIN behandlet_dokument bd ON bd.dokument_aar = btd.dokument_aar
        AND bd.dokument_nr = btd.dokument_nr
        AND bd.embete = btd.embete
    WHERE b.kansellert = false
      AND b.dato > '2026-07-24 14:04:00'
      AND bd.status = 'GODKJENT_MANUELT'
      AND bd.ml_behandlet IS NOT NULL
      AND bta.uthentet_versjon IS NOT NULL
    ORDER BY btd.dokument_aar, btd.dokument_nr, btd.embete, b.dato DESC
),
     per_dok AS (
         SELECT
             rb.dokument_aar,
             rb.dokument_nr,
             rb.embete,
             rb.dato,
             rb.rettsstiftelsestyper,
             COUNT(*) FILTER (WHERE l.ml_status IS DISTINCT FROM 'REJECTED'
                 AND l.id IS NOT NULL)                      AS vist,
             COUNT(*) FILTER (WHERE l.ml_generated = true
                 AND l.ml_status = 'ACCEPTED')              AS korrekt,
             COUNT(*) FILTER (WHERE l.ml_generated = true
                 AND l.ml_status = 'REJECTED')              AS oversladding,
             COUNT(*) FILTER (WHERE l.ml_generated = false)                AS manuelt
         FROM rb
                  LEFT JOIN label l ON l.dokument_aar = rb.dokument_aar
             AND l.dokument_nr = rb.dokument_nr
             AND l.embete = rb.embete
             AND l.fil_revisjon_id = rb.fil_revisjon_id
             AND l.versjon = rb.uthentet_versjon
             AND l.type = 'PERSONNUMMER'
         GROUP BY rb.dokument_aar, rb.dokument_nr, rb.embete, rb.dato, rb.rettsstiftelsestyper
     ),
     eksplodert AS (
         SELECT
             trim(unnest(string_to_array(rettsstiftelsestyper, ','))) AS rettsstiftelse,
             dato,
             vist,
             korrekt,
             oversladding,
             manuelt
         FROM per_dok
     ),
     rater AS (
         SELECT
             rettsstiftelse,
             GREATEST(MAX(dato)::date - MIN(dato)::date, 0) + 1              AS dager,
             COUNT(*)                                                         AS n_dok,
             SUM(korrekt)::numeric / COUNT(*)                                 AS korrekt_rate,
             SUM(oversladding)::numeric / COUNT(*)                            AS over_rate,
             SUM(manuelt)::numeric / COUNT(*)                                 AS manuelt_rate,
             SUM(vist)::numeric / COUNT(*)                                    AS vist_rate,
             SUM(korrekt)::numeric
                 / NULLIF(SUM(korrekt) + SUM(oversladding), 0)               AS presisjon,
             SUM(korrekt)::numeric / NULLIF(SUM(vist), 0)                    AS andel_av_vist,
             COUNT(*) FILTER (WHERE oversladding > 0)::numeric / COUNT(*)    AS dok_over_rate,
             COUNT(*) FILTER (WHERE manuelt > 0)::numeric / COUNT(*)         AS dok_manuelt_rate
         FROM eksplodert
         GROUP BY rettsstiftelse
     )
SELECT
    rettsstiftelse,
    dager                                                       AS dager_i_periode,
    n_dok                                                       AS antall_dokumenter,
    ROUND(korrekt_rate, 3)                                      AS korrekte_per_dok,
    ROUND(over_rate, 3)                                         AS oversladdinger_per_dok,
    ROUND(manuelt_rate, 3)                                      AS manuelt_tegnet_per_dok,
    ROUND(vist_rate, 3)                                         AS viste_sladdinger_per_dok,
    ROUND(100 * presisjon, 1)                                   AS ml_presisjon_prosent,
    ROUND(100 * andel_av_vist, 1)                               AS andel_ml_av_vist_prosent,
    ROUND(100 * dok_over_rate, 1)                               AS andel_dok_med_oversladding_prosent,
    ROUND(100 * dok_manuelt_rate, 1)                            AS andel_dok_med_manuelt_tegnet_prosent
FROM rater
ORDER BY n_dok DESC;

