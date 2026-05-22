-- Functions for dashboard statistics

-- Get disease statistics from results
CREATE OR REPLACE FUNCTION public.get_disease_statistics(time_range text DEFAULT 'all')
RETURNS TABLE (
  disease text,
  count bigint
) AS $$
DECLARE
  time_filter timestamp;
BEGIN
  -- Set time filter based on parameter
  IF time_range = 'month' THEN
    time_filter := now() - interval '1 month';
  ELSIF time_range = 'quarter' THEN
    time_filter := now() - interval '3 months';
  ELSIF time_range = 'year' THEN
    time_filter := now() - interval '1 year';
  ELSE
    time_filter := '1900-01-01'::timestamp; -- All time
  END IF;

  RETURN QUERY
  WITH user_analyses AS (
    -- Get analyses that belong to the user's patients
    SELECT a.id
    FROM analyses a
    JOIN patients p ON a.patient_id = p.id
    WHERE p.user_id = auth.uid()
    AND a.created_at >= time_filter
  ),
  predictions_flat AS (
    -- Flatten the predictions array from results
    SELECT r.id, 
           jsonb_array_elements(r.predictions) AS prediction
    FROM results r
    JOIN user_analyses ua ON r.analysis_id = ua.id
  ),
  disease_counts AS (
    -- Extract disease names and count them
    SELECT 
      prediction->>'disease' AS disease,
      COUNT(*) AS count
    FROM predictions_flat
    WHERE (prediction->>'probability')::float > 0.5 -- Only count significant findings
    GROUP BY prediction->>'disease'
    ORDER BY count DESC
  )
  SELECT disease, count FROM disease_counts;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Get analyses by month
CREATE OR REPLACE FUNCTION public.get_analyses_by_month(months_limit int DEFAULT 12)
RETURNS TABLE (
  month timestamp,
  count bigint
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    date_trunc('month', a.created_at) as month,
    COUNT(*) as count
  FROM analyses a
  JOIN patients p ON a.patient_id = p.id
  WHERE p.user_id = auth.uid()
    AND a.created_at >= now() - (months_limit || ' months')::interval
  GROUP BY date_trunc('month', a.created_at)
  ORDER BY month DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Get patient demographics by age group
CREATE OR REPLACE FUNCTION public.get_patient_demographics()
RETURNS TABLE (
  age_group text,
  count bigint
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    CASE
      WHEN age <= 18 THEN '0-18'
      WHEN age <= 35 THEN '19-35'
      WHEN age <= 50 THEN '36-50'
      WHEN age <= 65 THEN '51-65'
      ELSE '66+'
    END as age_group,
    COUNT(*) as count
  FROM patients
  WHERE user_id = auth.uid()
  GROUP BY age_group
  ORDER BY 
    CASE age_group
      WHEN '0-18' THEN 1
      WHEN '19-35' THEN 2
      WHEN '36-50' THEN 3
      WHEN '51-65' THEN 4
      WHEN '66+' THEN 5
    END;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER; 