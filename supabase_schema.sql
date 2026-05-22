-- Supabase SQL Schema for Chest X-ray Analysis System

-- Enable necessary extensions
create extension if not exists "uuid-ossp";
create extension if not exists "pgcrypto";

-- Set up storage buckets for X-ray images
insert into storage.buckets (id, name, public) 
values ('xray-images', 'xray-images', true);

-- Create storage policy to allow authenticated users to upload files
create policy "Allow authenticated users to upload X-ray images"
  on storage.objects for insert
  to authenticated
  with check (bucket_id = 'xray-images');

-- Create storage policy to allow public access to X-ray images
create policy "Allow public access to X-ray images"
  on storage.objects for select
  to public
  using (bucket_id = 'xray-images');

-- Create profiles table to store user profile information
create table public.profiles (
  id uuid references auth.users on delete cascade primary key,
  full_name text not null,
  phone_number text,
  specialty text,
  hospital text,
  created_at timestamptz default now() not null,
  updated_at timestamptz default now() not null
);

-- Set up RLS for profiles table
alter table public.profiles enable row level security;

-- Create policy to allow users to view their own profile
create policy "Users can view their own profile"
  on public.profiles for select
  to authenticated
  using (id = auth.uid());

-- Create policy to allow users to update their own profile
create policy "Users can update their own profile"
  on public.profiles for update
  to authenticated
  using (id = auth.uid());

-- Create patients table
create table public.patients (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid references auth.users not null,
  name text not null,
  age integer not null,
  gender text not null,
  medical_history text,
  notes text,
  created_at timestamptz default now() not null,
  updated_at timestamptz default now() not null
);

-- Set up RLS for patients table
alter table public.patients enable row level security;

-- Create policy to allow users to view their own patients
create policy "Users can view their own patients"
  on public.patients for select
  to authenticated
  using (user_id = auth.uid());

-- Create policy to allow users to insert their own patients
create policy "Users can insert their own patients"
  on public.patients for insert
  to authenticated
  with check (user_id = auth.uid());

-- Create policy to allow users to update their own patients
create policy "Users can update their own patients"
  on public.patients for update
  to authenticated
  using (user_id = auth.uid());

-- Create policy to allow users to delete their own patients
create policy "Users can delete their own patients"
  on public.patients for delete
  to authenticated
  using (user_id = auth.uid());

-- Create analyses table
create table public.analyses (
  id uuid primary key default uuid_generate_v4(),
  patient_id uuid references public.patients on delete cascade not null,
  image_url text not null,
  view_position text not null,
  notes text,
  created_at timestamptz default now() not null,
  updated_at timestamptz default now() not null
);

-- Set up RLS for analyses table
alter table public.analyses enable row level security;

-- Create policy to allow users to view analyses for their patients
create policy "Users can view analyses for their patients"
  on public.analyses for select
  to authenticated
  using (
    patient_id in (
      select id from public.patients where user_id = auth.uid()
    )
  );

-- Create policy to allow users to insert analyses for their patients
create policy "Users can insert analyses for their patients"
  on public.analyses for insert
  to authenticated
  with check (
    patient_id in (
      select id from public.patients where user_id = auth.uid()
    )
  );

-- Create policy to allow users to update analyses for their patients
create policy "Users can update analyses for their patients"
  on public.analyses for update
  to authenticated
  using (
    patient_id in (
      select id from public.patients where user_id = auth.uid()
    )
  );

-- Create policy to allow users to delete analyses for their patients
create policy "Users can delete analyses for their patients"
  on public.analyses for delete
  to authenticated
  using (
    patient_id in (
      select id from public.patients where user_id = auth.uid()
    )
  );

-- Create results table
create table public.results (
  id uuid primary key default uuid_generate_v4(),
  analysis_id uuid references public.analyses on delete cascade not null,
  predictions jsonb not null,
  heatmap_url text,
  doctor_notes text,
  created_at timestamptz default now() not null,
  updated_at timestamptz default now() not null
);

-- Set up RLS for results table
alter table public.results enable row level security;

-- Create policy to allow users to view results for their analyses
create policy "Users can view results for their analyses"
  on public.results for select
  to authenticated
  using (
    analysis_id in (
      select a.id from public.analyses a
      join public.patients p on a.patient_id = p.id
      where p.user_id = auth.uid()
    )
  );

-- Create policy to allow users to insert results for their analyses
create policy "Users can insert results for their analyses"
  on public.results for insert
  to authenticated
  with check (
    analysis_id in (
      select a.id from public.analyses a
      join public.patients p on a.patient_id = p.id
      where p.user_id = auth.uid()
    )
  );

-- Create policy to allow users to update results for their analyses
create policy "Users can update results for their analyses"
  on public.results for update
  to authenticated
  using (
    analysis_id in (
      select a.id from public.analyses a
      join public.patients p on a.patient_id = p.id
      where p.user_id = auth.uid()
    )
  );

-- Create policy to allow users to delete results for their analyses
create policy "Users can delete results for their analyses"
  on public.results for delete
  to authenticated
  using (
    analysis_id in (
      select a.id from public.analyses a
      join public.patients p on a.patient_id = p.id
      where p.user_id = auth.uid()
    )
  );

-- Create medical_conditions table to store disease classifications
create table public.medical_conditions (
  id uuid primary key default uuid_generate_v4(),
  name text not null unique,
  description text,
  created_at timestamptz default now() not null
);

-- Make medical_conditions accessible to all authenticated users
alter table public.medical_conditions enable row level security;
create policy "Medical conditions are viewable by all authenticated users"
  on public.medical_conditions for select
  to authenticated
  using (true);

-- Populate medical_conditions with common chest X-ray findings
insert into public.medical_conditions (name, description) values
  ('Atelectasis', 'Collapse of lung tissue affecting part or all of one lung'),
  ('Cardiomegaly', 'Enlargement of the heart'),
  ('Consolidation', 'Lung tissue filled with liquid instead of air'),
  ('Edema', 'Buildup of fluid in the lungs'),
  ('Effusion', 'Buildup of fluid between the layers of tissue that line the lungs and chest cavity'),
  ('Emphysema', 'Condition in which the air sacs of the lungs are damaged and enlarged'),
  ('Fibrosis', 'Scarring of lung tissue'),
  ('Hernia', 'Protrusion of organs through the diaphragm'),
  ('Infiltration', 'Abnormal substances have infiltrated the lung'),
  ('Mass', 'Abnormal spot or growth in the lung'),
  ('Nodule', 'Small round growth in the lung'),
  ('Pleural Thickening', 'Thickening of the pleural space'),
  ('Pneumonia', 'Infection that inflames air sacs in one or both lungs'),
  ('Pneumothorax', 'Collapsed lung due to air in the pleural space'),
  ('Tuberculosis', 'Infectious disease that primarily affects the lungs');

-- Create a function to check if a user owns a patient
create or replace function public.user_owns_patient(patient_id uuid)
returns boolean as $$
begin
  return exists (
    select 1 from public.patients
    where id = patient_id and user_id = auth.uid()
  );
end;
$$ language plpgsql security definer;

-- Create a function to get patient statistics for a user
create or replace function public.get_user_statistics()
returns jsonb as $$
declare
  result jsonb;
begin
  select json_build_object(
    'total_patients', (select count(*) from public.patients where user_id = auth.uid()),
    'total_analyses', (
      select count(*) from public.analyses a
      join public.patients p on a.patient_id = p.id
      where p.user_id = auth.uid()
    ),
    'analyses_by_month', (
      select json_agg(monthly)
      from (
        select 
          date_trunc('month', a.created_at) as month,
          count(*) as count
        from public.analyses a
        join public.patients p on a.patient_id = p.id
        where p.user_id = auth.uid()
        group by date_trunc('month', a.created_at)
        order by date_trunc('month', a.created_at) desc
        limit 12
      ) as monthly
    )
  ) into result;
  
  return result;
end;
$$ language plpgsql security definer;

-- Create triggers for updated_at columns
create or replace function public.handle_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

-- Create triggers for each table with updated_at column
create trigger handle_updated_at_profiles
  before update on public.profiles
  for each row execute procedure public.handle_updated_at();

create trigger handle_updated_at_patients
  before update on public.patients
  for each row execute procedure public.handle_updated_at();

create trigger handle_updated_at_analyses
  before update on public.analyses
  for each row execute procedure public.handle_updated_at();

create trigger handle_updated_at_results
  before update on public.results
  for each row execute procedure public.handle_updated_at();

-- Create a view to simplify querying patient data with analysis counts
create or replace view public.patient_summaries as
select
  p.id,
  p.name,
  p.age,
  p.gender,
  p.created_at,
  p.user_id,
  (
    select count(*)
    from public.analyses a
    where a.patient_id = p.id
  ) as analyses_count,
  (
    select max(a.created_at)
    from public.analyses a
    where a.patient_id = p.id
  ) as latest_analysis_date
from public.patients p;

-- Instead of RLS on view, create a secure function to access patient summaries
create or replace function public.get_my_patient_summaries()
returns setof public.patient_summaries as $$
begin
  return query
  select * from public.patient_summaries
  where user_id = auth.uid();
end;
$$ language plpgsql security definer; 