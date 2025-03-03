"""
Summary of all tags and categories used in the YouTube Content Analysis
"""

YOUTUBE_CATEGORIES = {
    "primary_categories": [
        "Music", "Gaming", "Education", "Entertainment", "Sports",
        "News", "Technology", "Comedy", "Vlogs", "Tutorial",
        "Reviews", "Podcast", "Shorts"
    ],
    
    "detailed_categories": {
        "Music": [
            "Official Music Video", "Live Performance", "Cover Song",
            "Lyric Video", "Music Mix", "Concert Recording",
            "Music Review", "Behind the Scenes"
        ],
        "Gaming": [
            "Gameplay", "Tutorial", "Review", "Esports",
            "Walkthrough", "Gaming News", "Stream Highlights"
        ],
        "Education": [
            "Tutorial", "Lecture", "How-to Guide", "Educational Series",
            "Documentary", "Course Material"
        ]
    }
}

MUSIC_SPECIFIC = {
    'mood': [
        'happy', 'sad', 'energetic', 'calm', 'aggressive', 
        'relaxed', 'dark', 'atmospheric'
    ],
    'genre': [
        'electronic', 'rock', 'pop', 'hip hop', 'classical',
        'jazz', 'metal', 'folk', 'ambient', 'indie'
    ],
    'characteristics': [
        'instrumental', 'vocal', 'fast', 'slow', 'melodic',
        'rhythmic', 'acoustic', 'synthetic'
    ],
    'context': [
        'party', 'study', 'workout', 'sleep', 'meditation',
        'dance', 'focus', 'background'
    ]
}

MENTAL_HEALTH_TAGS = {
    'emotional_states': {
        'anxiety': [
            'anxiety', 'stress', 'worried', 'panic', 'overthinking',
            'nervous', 'fear', 'tension', 'anxious', 'paranoid',
            'phobia', 'scared', 'dread', 'uneasy', 'restless'
        ],
        'depression': [
            'depression', 'sad', 'lonely', 'hopeless', 'down',
            'blue', 'melancholy', 'grief', 'despair', 'worthless',
            'numb', 'empty', 'crying', 'suicidal', 'depressed'
        ]
    },
    
    'addictive_behaviors': {
        'general_addiction': [
            'addiction', 'cant stop', 'binge', 'compulsive',
            'obsessed', 'hooked', 'dependent', 'craving', 'withdrawal',
            'relapse', 'recovery', 'sober', 'clean', 'urge', 'temptation'
        ],
        'gaming_addiction': [
            'gaming addiction', 'game addiction', 'addicted to games',
            'gaming disorder', 'excessive gaming', 'gaming problem',
            'cant stop playing', 'gaming withdrawal'
        ],
        'gambling': [
            'gambling', 'betting', 'casino', 'slots', 'poker',
            'lottery', 'wagering', 'stake', 'odds', 'jackpot',
            'gambling addiction', 'gambling problem'
        ],
        'social_media_addiction': [
            'social media addiction', 'phone addiction', 'screen time',
            'digital detox', 'internet addiction', 'scrolling',
            'always online', 'cant disconnect'
        ]
    },
    
    'behavioral_patterns': {
        'impulsivity': [
            'impulsive', 'reckless', 'spontaneous', 'risky behavior',
            'poor judgment', 'acting without thinking', 'impulse control'
        ],
        'procrastination': [
            'procrastination', 'avoiding', 'putting off', 'delay',
            'distraction', 'time management', 'productivity struggle'
        ],
        'perfectionism': [
            'perfectionist', 'perfect', 'not good enough', 'high standards',
            'overachiever', 'self-critical', 'demanding'
        ]
    },
    
    'relationship_patterns': {
        'relationship_issues': [
            'relationship problems', 'trust issues', 'attachment',
            'codependency', 'toxic relationship', 'boundaries',
            'relationship anxiety', 'abandonment'
        ],
        'social_anxiety': [
            'social anxiety', 'shy', 'introvert', 'social fear',
            'social phobia', 'social isolation', 'avoiding people'
        ]
    },
    
    'coping_mechanisms': {
        'avoidance': [
            'avoiding', 'escape', 'running away', 'hiding',
            'withdrawal', 'isolation', 'shutting down'
        ],
        'self_medication': [
            'self medication', 'coping', 'substance use',
            'drinking to cope', 'using to cope', 'numbing'
        ]
    }
}

CONTENT_ANALYSIS = {
    'content_format': [
        'tutorial', 'review', 'reaction', 'compilation', 'highlights',
        'commentary', 'analysis', 'walkthrough', 'guide', 'showcase',
        'stream_highlights', 'podcast', 'interview', 'documentary'
    ],
    'production_quality': [
        'professional', 'amateur', 'high_production', 'live_recording',
        'studio_quality', 'raw_footage', 'edited', 'animated'
    ],
    'audience_engagement': [
        'educational', 'entertaining', 'informative', 'controversial',
        'clickbait', 'viral', 'trending', 'family_friendly', 'mature'
    ],
    'content_purpose': [
        'skill_development', 'entertainment', 'news', 'analysis',
        'social_commentary', 'product_review', 'storytelling',
        'educational', 'promotional', 'personal_vlog'
    ]
}

PATTERN_INDICATORS = {
    'series': [
        r'part\s*\d+', r'episode\s*\d+', r'ep\.\s*\d+',
        r'#\d+', r'\[\d+\]', r'series', r'season'
    ],
    'trending': [
        r'trend(?:ing)?', r'viral', r'challenge', r'tiktok',
        r'shorts', r'latest', r'new'
    ],
    'clickbait': [
        r'you won\'?t believe', r'shocking', r'amazing',
        r'\(\d+\s*shocking.*\)', r'gone wrong', r'mind blown'
    ],
    'monetization': [
        r'sponsored', r'ad\b', r'#ad', r'partner',
        r'affiliate', r'promo'
    ],
    'community': [
        r'collab', r'ft\.', r'feat', r'featuring',
        r'w/', r'with', r'vs'
    ]
}

GENERAL_CONTENT = {
    'format': [
        'tutorial', 'review', 'gameplay', 'vlog', 
        'reaction', 'commentary', 'guide', 'news'
    ],
    'purpose': [
        'entertainment', 'education', 'information',
        'comedy', 'storytelling', 'analysis'
    ],
    'style': [
        'casual', 'professional', 'funny', 'serious',
        'dramatic', 'informative', 'personal'
    ],
    'audience': [
        'gamers', 'students', 'professionals', 'general',
        'kids', 'teens', 'adults'
    ]
} 