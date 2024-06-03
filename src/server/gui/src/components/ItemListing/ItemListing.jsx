import React, { useState } from "react";
import { Container, Typography, Tabs, Tab, Card, CardContent, CardMedia, Grid, useMediaQuery, useTheme, Box } from '@mui/material';

export default function AgentsListing() {
  const [value, setValue] = useState(0);
  const [category] = useState(["Writing", "Productivity", "Research & Analysis"]);
  const [items] = useState([
    { 'name': '数学计算智能体', 'planning': 'ReAct Planning', 'tools': '加法', 'description': '数学计算智能体可以计算加减乘除，开平方根', 'llm': 'gpt4', 'image': '/path/to/image1.jpg' },
    { 'name': '数据库智能体', 'planning': 'ReAct Planning', 'tools': '数据库操作tools', 'description': '调用数据库，完成text2sql任务', 'llm': 'gpt4', 'image': '/path/to/image2.jpg' }
  ]);

  const handleChange = (event, newValue) => {
    setValue(newValue);
  };

  const theme = useTheme();
  const isLargeScreen = useMediaQuery('(min-width:700px)');

  return (
    <Container>
      <Tabs
        value={value}
        onChange={handleChange}
        aria-label="tabs"
        TabIndicatorProps={{ style: { backgroundColor: 'black' } }}
      >
        {category.map((item, index) => (
          <Tab 
            key={index} 
            label={item} 
            sx={{
              color: value === index ? 'black' : 'grey',
              '&.Mui-selected': { color: 'black' }
            }}
          />
        ))}
      </Tabs>
      <Box mt={3}>
        <Grid container spacing={3}>
          {items.map((item, index) => (
            <Grid item xs={12} sm={isLargeScreen ? 6 : 12} md={isLargeScreen ? 4 : 12} key={index}>
              <Card sx={{ maxWidth: 345, boxShadow: 3, minHeight:200, backgroundColor: 'white' }}>
                <CardContent>
                  <Typography variant="h6" component="div" gutterBottom color="textPrimary" sx={{ color: 'black' }}>
                    {item.name}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" component="p" gutterBottom sx={{ color: 'grey.800' }}>
                    <strong>决策框架：</strong>{item.planning}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" component="p" gutterBottom sx={{ color: 'grey.800' }}>
                    <strong>工具库：</strong>{item.tools}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" component="p" gutterBottom sx={{ color: 'grey.800' }}>
                    <strong>使用底座大模型：</strong>{item.llm}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" component="p" gutterBottom sx={{ color: 'grey.800' }}>
                    <strong>描述：</strong>{item.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    </Container>
  );
}