import React, { useState } from 'react';
import { Button, TextField, Select, MenuItem, FormControl, InputLabel, Box, Typography, Paper } from '@mui/material';

// Define sequence types
const SEQUENCE_TYPES = {
    INTRODUCTION: "Opening introduction and hook",
    VALUE_PROP: "Company and role value proposition",
    EXPERIENCE_MATCH: "Candidate experience and role match",
    CALL_TO_ACTION: "Call to action and next steps",
    CLOSING: "Professional closing"
};

interface SequenceContext {
    candidateName?: string;
    role?: string;
    company?: string;
    candidateBackground?: string;
    [key: string]: string | undefined;
}

interface SequenceGeneratorProps {
    backendUrl: string;
    onMessageGenerated?: (message: string) => void;
}

const SequenceGenerator: React.FC<SequenceGeneratorProps> = ({ backendUrl, onMessageGenerated }) => {
    const [sequenceType, setSequenceType] = useState<string>('');
    const [context, setContext] = useState<SequenceContext>({
        candidateName: '',
        role: '',
        company: '',
        candidateBackground: ''
    });
    const [generatedContent, setGeneratedContent] = useState<string>('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleContextChange = (field: keyof SequenceContext) => (event: React.ChangeEvent<HTMLInputElement>) => {
        setContext(prev => ({
            ...prev,
            [field]: event.target.value
        }));
    };

    const generateSequencePart = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${backendUrl}/api/generate_sequence_part`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    sequence_type: sequenceType,
                    context: context
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setGeneratedContent(data.result);
            if (onMessageGenerated) {
                onMessageGenerated(data.result);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    const generateFullSequence = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${backendUrl}/api/generate_full_sequence`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    context: context
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const fullMessage = data.full_message;
            setGeneratedContent(fullMessage);
            if (onMessageGenerated) {
                onMessageGenerated(fullMessage);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    return (
        <Paper elevation={3} sx={{ p: 3, maxWidth: '800px', margin: '20px auto' }}>
            <Typography variant="h6" gutterBottom>
                Sequence Message Generator
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                    label="Candidate Name"
                    value={context.candidateName}
                    onChange={handleContextChange('candidateName')}
                    fullWidth
                />
                
                <TextField
                    label="Role"
                    value={context.role}
                    onChange={handleContextChange('role')}
                    fullWidth
                />
                
                <TextField
                    label="Company"
                    value={context.company}
                    onChange={handleContextChange('company')}
                    fullWidth
                />
                
                <TextField
                    label="Candidate Background"
                    value={context.candidateBackground}
                    onChange={handleContextChange('candidateBackground')}
                    multiline
                    rows={3}
                    fullWidth
                />

                <FormControl fullWidth>
                    <InputLabel>Sequence Type</InputLabel>
                    <Select
                        value={sequenceType}
                        label="Sequence Type"
                        onChange={(e) => setSequenceType(e.target.value)}
                    >
                        {Object.entries(SEQUENCE_TYPES).map(([key, value]) => (
                            <MenuItem key={key} value={key}>{value}</MenuItem>
                        ))}
                    </Select>
                </FormControl>

                <Box sx={{ display: 'flex', gap: 2, justifyContent: 'space-between' }}>
                    <Button
                        variant="contained"
                        onClick={generateSequencePart}
                        disabled={loading || !sequenceType}
                        sx={{ flex: 1 }}
                    >
                        Generate Sequence Part
                    </Button>
                    <Button
                        variant="contained"
                        onClick={generateFullSequence}
                        disabled={loading}
                        sx={{ flex: 1 }}
                    >
                        Generate Full Sequence
                    </Button>
                </Box>

                {error && (
                    <Typography color="error" variant="body2">
                        {error}
                    </Typography>
                )}

                {generatedContent && (
                    <TextField
                        label="Generated Content"
                        value={generatedContent}
                        multiline
                        rows={6}
                        fullWidth
                        InputProps={{
                            readOnly: true,
                        }}
                    />
                )}
            </Box>
        </Paper>
    );
};

export default SequenceGenerator; 