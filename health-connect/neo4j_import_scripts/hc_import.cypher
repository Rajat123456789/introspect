// 0. Delete all nodes
MATCH (n)
DETACH DELETE n;

// 1. Ensure the User node exists (using parameterized username)
MERGE (u:User { username: $username });

// Basal Metabolic Rate
CALL apoc.file.exists("file:///basalMetabolicRate_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///basalMetabolicRate_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_BASAL_METABOLIC_RATE]->(bmr:BasalMetabolicRate {
      metricName: "Basal Metabolic Rate",
      start: row.start,
      basalMetabolicRate_inKilocaloriesPerDay_value: toFloat(row.basalMetabolicRate_inKilocaloriesPerDay),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Body Fat
CALL apoc.file.exists("file:///bodyFat_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///bodyFat_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_BODY_FAT]->(bf:BodyFat {
      metricName: "Body Fat",
      start: row.start,
      bodyFat_percentage_value: toFloat(row.bodyFat_percentage),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Distance
CALL apoc.file.exists("file:///distance_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///distance_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_DISTANCE]->(d:Distance {
      metricName: "Distance",
      start: row.start,
      end: row.end,
      distance_inKilometers_value: toFloat(row.distance_inKilometers),
      distance_inMiles_value: toFloat(row.distance_inMiles),
      distance_total_time_value: toFloat(row.distance_total_time),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Elevation Gained
CALL apoc.file.exists("file:///elevationGained_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///elevationGained_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime, datetime(replace(row.end, " ", "T")) AS endTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_ELEVATION_GAINED]->(eg:ElevationGained {
      metricName: "Elevation Gained",
      start: startTime,
      end: endTime,
      elevationGained_elevation_inFeet_value: toFloat(row.elevationGained_elevation_inFeet),
      elevationGained_elevation_inMeters_value: toFloat(row.elevationGained_elevation_inMeters),
      elevationGained_total_time_value: toFloat(row.elevationGained_total_time),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Exercise Session
CALL apoc.file.exists("file:///exerciseSession_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///exerciseSession_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_EXERCISE_SESSION]->(es:ExerciseSession {
      metricName: "Exercise Session",
      start: row.start,
      end: row.end,
      exerciseSession_endZoneOffset_id_value: row.exerciseSession_endZoneOffset_id,
      exerciseSession_endZoneOffset_totalSeconds_value: toInteger(row.exerciseSession_endZoneOffset_totalSeconds),
      exerciseSession_exerciseType_value: toInteger(row.exerciseSession_exerciseType),
      exerciseSession_laps_value: row.exerciseSession_laps,
      exerciseSession_notes_value: row.exerciseSession_notes,
      exerciseSession_segments_value: row.exerciseSession_segments,
      exerciseSession_startZoneOffset_id_value: row.exerciseSession_startZoneOffset_id,
      exerciseSession_startZoneOffset_totalSeconds_value: row.exerciseSession_startZoneOffset_totalSeconds,
      exerciseSession_title_value: row.exerciseSession_title,
      exerciseSession_total_time: row.exerciseSession_total_time
  });
  ", "", {username:$username}) YIELD value;

// Floors Climbed
CALL apoc.file.exists("file:///floorsClimbed_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///floorsClimbed_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime, datetime(replace(row.end, " ", "T")) AS endTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_FLOORS_CLIMBED]->(fc:FloorsClimbed {
      metricName: "Floors Climbed",
      start: startTime,
      end: endTime,
      floorsClimbed_floors_value: toInteger(row.floorsClimbed_floors),
      floorsClimbed_total_time_value: toFloat(row.floorsClimbed_total_time),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Heart Rate
CALL apoc.file.exists("file:///heartRate_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///heartRate_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_HEART_RATE]->(hr:HeartRate {
      metricName: "Heart Rate",
      start: row.start,
      beatsPerMinute_value: toInteger(row.beatsPerMinute),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Height
CALL apoc.file.exists("file:///height_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///height_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_HEIGHT]->(h:Height {
      metricName: "Height",
      start: row.start,
      height_inFeet_value: toFloat(row.height_inFeet),
      height_inInches_value: toFloat(row.height_inInches),
      height_inMeters_value: toFloat(row.height_inMeters),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Nutrition (calories, fat, carbs, protein)
CALL apoc.file.exists("file:///nutrition_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///nutrition_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_NUTRITION]->(n:Nutrition {
      metricName: "Nutrition",
      start: startTime,
      calories_value: toInteger(row.calories),
      fat_inGrams_value: toFloat(row.fat_inGrams),
      carbs_inGrams_value: toFloat(row.carbs_inGrams),
      protein_inGrams_value: toFloat(row.protein_inGrams),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Nutrition (detailed nutrient values)
CALL apoc.file.exists("file:///nutrition_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///nutrition_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_NUTRITION]->(n:Nutrition {
      metricName: "Nutrition",
      start: row.start,
      biotin_inGrams_value: toFloat(row.biotin_inGrams),
      caffeine_inGrams_value: toFloat(row.caffeine_inGrams),
      calcium_inGrams_value: toFloat(row.calcium_inGrams),
      chloride_inGrams_value: toFloat(row.chloride_inGrams),
      cholesterol_inGrams_value: toFloat(row.cholesterol_inGrams),
      chromium_inGrams_value: toFloat(row.chromium_inGrams),
      copper_inGrams_value: toFloat(row.copper_inGrams),
      dietaryFiber_inGrams_value: toFloat(row.dietaryFiber_inGrams),
      folate_inGrams_value: toFloat(row.folate_inGrams),
      folicAcid_inGrams_value: toFloat(row.folicAcid_inGrams),
      iodine_inGrams_value: toFloat(row.iodine_inGrams),
      iron_inGrams_value: toFloat(row.iron_inGrams),
      magnesium_inGrams_value: toFloat(row.magnesium_inGrams),
      manganese_inGrams_value: toFloat(row.manganese_inGrams),
      molybdenum_inGrams_value: toFloat(row.molybdenum_inGrams),
      monounsaturatedFat_inGrams_value: toFloat(row.monounsaturatedFat_inGrams),
      niacin_inGrams_value: toFloat(row.niacin_inGrams),
      pantothenicAcid_inGrams_value: toFloat(row.pantothenicAcid_inGrams),
      phosphorus_inGrams_value: toFloat(row.phosphorus_inGrams),
      polyunsaturatedFat_inGrams_value: toFloat(row.polyunsaturatedFat_inGrams),
      potassium_inGrams_value: toFloat(row.potassium_inGrams),
      protein_inGrams_value: toFloat(row.protein_inGrams),
      riboflavin_inGrams_value: toFloat(row.riboflavin_inGrams),
      saturatedFat_inGrams_value: toFloat(row.saturatedFat_inGrams),
      selenium_inGrams_value: toFloat(row.selenium_inGrams),
      sodium_inGrams_value: toFloat(row.sodium_inGrams),
      sugar_inGrams_value: toFloat(row.sugar_inGrams),
      thiamin_inGrams_value: toFloat(row.thiamin_inGrams),
      totalCarbohydrate_inGrams_value: toFloat(row.totalCarbohydrate_inGrams),
      totalFat_inGrams_value: toFloat(row.totalFat_inGrams),
      transFat_inGrams_value: toFloat(row.transFat_inGrams),
      unsaturatedFat_inGrams_value: toFloat(row.unsaturatedFat_inGrams),
      vitaminA_inGrams_value: toFloat(row.vitaminA_inGrams),
      vitaminB12_inGrams_value: toFloat(row.vitaminB12_inGrams),
      vitaminB6_inGrams_value: toFloat(row.vitaminB6_inGrams),
      vitaminC_inGrams_value: toFloat(row.vitaminC_inGrams),
      vitaminD_inGrams_value: toFloat(row.vitaminD_inGrams),
      vitaminE_inGrams_value: toFloat(row.vitaminE_inGrams),
      vitaminK_inGrams_value: toFloat(row.vitaminK_inGrams),
      zinc_inGrams_value: toFloat(row.zinc_inGrams),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Oxygen Saturation          
CALL apoc.file.exists("file:///oxygenSaturation_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///oxygenSaturation_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_OXYGEN_SATURATION]->(os:OxygenSaturation {
      metricName: "Oxygen Saturation",
      start: row.start,
      oxygenSaturation_percentage_value: toInteger(row.oxygenSaturation_percentage),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Sleep Session
CALL apoc.file.exists("file:///sleepSession_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///sleepSession_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime, datetime(replace(row.end, " ", "T")) AS endTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_SLEEP_SESSION]->(ss:SleepSession {
      metricName: "Sleep Session",
      start: row.start,
      end: row.end,
      sleep_stage_1_value: toFloat(row.sleep_stage_1),
      sleep_stage_2_value: toFloat(row.sleep_stage_2),
      sleep_stage_3_value: toFloat(row.sleep_stage_3),
      sleep_stage_4_value: toFloat(row.sleep_stage_4),
      sleep_stage_5_value: toFloat(row.sleep_stage_5),
      sleep_stage_6_value: toFloat(row.sleep_stage_6),
      sleep_stage_7_value: toFloat(row.sleep_stage_7),
      sleep_stage_8_value: toFloat(row.sleep_stage_8),
      total_sleep_time_value: toFloat(row.total_sleep_time),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Speed            
CALL apoc.file.exists("file:///speed_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///speed_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_SPEED]->(s:Speed {
      metricName: "Speed",
      start: row.start,
      end: row.end,
      speed_total_time_spent_value: toFloat(row.speed_total_time_spent),
      average_speed_kmh_value: toFloat(row.average_speed_kmh),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Steps
CALL apoc.file.exists("file:///steps_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///steps_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_STEPS]->(st:Steps {
      metricName: "Steps",
      start: row.start,
      steps_count_value: toInteger(row.steps_count),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Total Calories Burned                
CALL apoc.file.exists("file:///totalCaloriesBurned_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///totalCaloriesBurned_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime, datetime(replace(row.end, " ", "T")) AS endTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_TOTAL_CALORIES_BURNED]->(tc:TotalCaloriesBurned {
      metricName: "Total Calories Burned",
      start: row.start,
      end: row.end,
      totalCaloriesBurned_energy_inKilocalories_value: toFloat(row.totalCaloriesBurned_energy_inKilocalories),
      totalCaloriesBurned_total_time_value: toInteger(row.totalCaloriesBurned_total_time),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Weight
CALL apoc.file.exists("file:///weight_" + $username + "_Cleaned.csv") YIELD value AS fileExists
CALL apoc.do.when(fileExists, 
  "
  LOAD CSV WITH HEADERS FROM "file:///weight_" + $username + "_Cleaned.csv" AS row
  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
  MATCH (u:User {username: $username})
  CREATE (u)-[:HAS_WEIGHT]->(w:Weight {
      metricName: "Weight",
      start: row.start,
      weight_inKilograms_value: toFloat(row.weight_inKilograms),
      weight_inPounds_value: toFloat(row.weight_inPounds),
      appName: row.app
  });
  ", "", {username:$username}) YIELD value;

// Final return
RETURN "Data import complete" AS result;
