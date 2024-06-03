import React, { useState } from 'react';
import { List, ListItemButton, ListItemIcon, ListItemText, Collapse, Typography } from '@mui/material';
import ExpandLess from '@mui/icons-material/ExpandLess';
import ExpandMore from '@mui/icons-material/ExpandMore';
import { PeopleAltOutlined, ConstructionOutlined, AssignmentOutlined, GroupsOutlined, FactoryOutlined } from '@mui/icons-material';
import { grey } from '@mui/material/colors';

const SIDES_LISTS_GENERAL = {
  'Agent智能体': <PeopleAltOutlined />,
  '工具库': <ConstructionOutlined />,
  '任务': <AssignmentOutlined />,
  '团队': <GroupsOutlined />,
  '工作间': <FactoryOutlined />,
};

export default function NestedList() {
  const [open, setOpen] = useState(true);

  const handleClick = () => {
    setOpen(!open);
  };

  return (
    <List
      sx={{
        width: '100%',
        maxWidth: 240,
        bgcolor: "#f5f6fe",
        borderRadius: 2,
        boxShadow: 3,
        overflow: 'hidden',
      }}
    >
      <ListItemButton onClick={handleClick} sx={{ bgcolor: "#f5f6fe", color: 'text.primary' }}>
        <ListItemText primary={<Typography variant="h6">我的</Typography>} />
        {open ? <ExpandLess /> : <ExpandMore />}
      </ListItemButton>

      <Collapse in={open} timeout="auto" unmountOnExit>
        <List component="div" disablePadding>
          {Object.keys(SIDES_LISTS_GENERAL).map((item) => (
            <ListItemButton key={item} sx={{ pl: 3, bgcolor: "#f5f6fe", color: 'text.primary' }}>
              <ListItemIcon sx={{ minWidth: 30, color: 'text.primary' }}>{SIDES_LISTS_GENERAL[item]}</ListItemIcon>
              <ListItemText primary={<Typography variant="body1">{item}</Typography>} sx={{ ml: 0 }} />
            </ListItemButton>
          ))}
        </List>
      </Collapse>
    </List>
  );
}