// 1. Ensure the User node exists
MERGE (u:User { username: "someshbgd3" });

// 2. Import Heart Rate Measurements
LOAD CSV WITH HEADERS FROM 'file:///Cleaned_heartRate_someshbgd3.csv' AS row
WITH row, datetime(replace(row.start, " ", "T")) AS startTime
MATCH (u:User {username: "someshbgd3"})
CREATE (u)-[:HAS_HEART_RATE]->(hr:HeartRate {
    start: startTime,
    value: toInteger(row.beatsPerMinute),
    appName: row.app
});

// 3. Import Speed Measurements
LOAD CSV WITH HEADERS FROM 'file:///Cleaned_speed_someshbgd3.csv' AS row
WITH row, datetime(replace(row.start, " ", "T")) AS startTime
MATCH (u:User {username: "someshbgd3"})
CREATE (u)-[:HAS_SPEED]->(s:Speed {
  start: startTime,
  end: row.end
  speed: toFloat(row.average_speed_kmh),
  appName: row.app
});

    
// 4. Import Sleep Session Data
    LOAD CSV ///Cleaned_sleepSession_someshbgd3.csv' AS row
    WITH HEADERS FROM 'file:
    WITH row, datetime(replace(row.start, " ", "T")) AS startTime, datetime(replace(row.end, " ", "T")) AS endTime
    MATCH (u:User { username: "someshbgd3" })
    CREATE (u)-[:HAS_SLEEP_SESSION]->(ss:SleepSession {
      start: startTime,
      end: endTime,
      duration: toInteger(row.duration),
      appName: row.appName
      });
      
// 5. Import Nutrition Data
      LOAD CSV ///Cleaned_nutrition_someshbgd3.csv' AS row
      WITH HEADERS FROM 'file:
      WITH row, datetime(replace(row.start, " ", "T")) AS startTime
      MATCH (u:User { username: "someshbgd3" })
      CREATE (u)-[:HAS_NUTRITION]->(n:Nutrition {
        start: startTime,
        calories: toInteger(row.calories),
        protein: toFloat(row.protein),
        carbs: toFloat(row.carbs),
        fat: toFloat(row.fat),
        appName: row.appName
        });
        
// 6. Import Exercise Session Data
        LOAD CSV ///Cleaned_exerciseSession_someshbgd3.csv' AS row
        WITH HEADERS FROM 'file:
        WITH row, datetime(replace(row.start, " ", "T")) AS startTime
        MATCH (u:User { username: "someshbgd3" })
        CREATE (u)-[:HAS_EXERCISE_SESSION]->(ex:ExerciseSession {
          start: startTime,
          duration: toInteger(row.duration),
          type: row.type,
          appName: row.appName
          });
          
// 7. Import Total Calories Burned Data
          LOAD CSV ///Cleaned_totalCaloriesBurned_someshbgd3.csv' AS row
          WITH HEADERS FROM 'file:
          WITH row, datetime(replace(row.start, " ", "T")) AS startTime
          MATCH (u:User { username: "someshbgd3" })
          CREATE (u)-[:HAS_TOTAL_CALORIES_BURNED]->(tc:TotalCaloriesBurned {
            start: startTime,
            value: toInteger(row.value),
            appName: row.appName
            });
            
// 8. Import Height Data
            LOAD CSV ///Cleaned_height_someshbgd3.csv' AS row
            WITH HEADERS FROM 'file:
            WITH row, datetime(replace(row.start, " ", "T")) AS startTime
            MATCH (u:User { username: "someshbgd3" })
            CREATE (u)-[:HAS_HEIGHT]->(h:Height {
              start: startTime,
              value: toFloat(row.value),
              appName: row.appName
              });
              
// 9. Import Weight Data
              LOAD CSV ///Cleaned_weight_someshbgd3.csv' AS row
              WITH HEADERS FROM 'file:
              WITH row, datetime(replace(row.start, " ", "T")) AS startTime
              MATCH (u:User { username: "someshbgd3" })
              CREATE (u)-[:HAS_WEIGHT]->(w:Weight {
                start: startTime,
                value: toFloat(row.value),
                appName: row.appName
                });
                
// 10. Import Steps Data
                LOAD CSV ///Cleaned_steps_someshbgd3.csv' AS row
                WITH HEADERS FROM 'file:
                WITH row, datetime(replace(row.start, " ", "T")) AS startTime
                MATCH (u:User { username: "someshbgd3" })
                CREATE (u)-[:HAS_STEPS]->(st:Steps {
                  start: startTime,
                  count: toInteger(row.count),
                  appName: row.appName
                  });
                  
// 11. Import Distance Data
                  LOAD CSV ///Cleaned_distance_someshbgd3.csv' AS row
                  WITH HEADERS FROM 'file:
                  WITH row, datetime(replace(row.start, " ", "T")) AS startTime
                  MATCH (u:User { username: "someshbgd3" })
                  CREATE (u)-[:HAS_DISTANCE]->(d:Distance {
                    start: startTime,
                    value: toFloat(row.value),
                    appName: row.appName
                    });
                    
// 12. Import Oxygen Saturation Data
                    LOAD CSV ///Cleaned_oxygenSaturation_someshbgd3.csv' AS row
                    WITH HEADERS FROM 'file:
                    WITH row, datetime(replace(row.start, " ", "T")) AS startTime
                    MATCH (u:User { username: "someshbgd3" })
                    CREATE (u)-[:HAS_OXYGEN_SATURATION]->(os:OxygenSaturation {
                      start: startTime,
                      value: toFloat(row.value), // assuming oxygen saturation is a float percentage (e.g., 97.5)
                      appName: row.appName
                      });
                      
// 13. Import Body Fat Data
                      LOAD CSV ///Cleaned_bodyFat_someshbgd3.csv' AS row
                      WITH HEADERS FROM 'file:
                      WITH row, datetime(replace(row.start, " ", "T")) AS startTime
                      MATCH (u:User { username: "someshbgd3" })
                      CREATE (u)-[:HAS_BODY_FAT]->(bf:BodyFat {
                        start: startTime,
                        value: toFloat(row.value), // adjust if your CSV uses a different header for the body fat measurement
                        appName: row.appName
                        });
                        
// 14. Import Basal Metabolic Rate Data
                        LOAD CSV ///Cleaned_basalMetabolicRate_someshbgd3.csv' AS row
                        WITH HEADERS FROM 'file:
                        WITH row, datetime(replace(row.start, " ", "T")) AS startTime
                        MATCH (u:User { username: "someshbgd3" })
                        CREATE (u)-[:HAS_BASAL_METABOLIC_RATE]->(bmr:BasalMetabolicRate {
                          start: startTime,
                          value: toInteger(row.value), // adjust data type if BMR is stored as an integer (calories) or float
                          appName: row.appName
                          });
