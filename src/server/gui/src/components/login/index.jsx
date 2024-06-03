import React from 'react'
import { Grid,Paper, Avatar, TextField, Button, Typography, Link} from '@mui/material';
import { LockOutlined } from '@mui/icons-material';

const LoginModule=()=>{
    const paperStyle={padding :20,height:"max-content",width:280, margin:"20px auto"}
    const avatarStyle={backgroundColor:'#1bbd7e'}
    const btnstyle={margin:'8px 0'}

    return(
        <Grid>
            <Paper elevation={10} style={paperStyle}>
                <Grid align='center'>
                     <Avatar style={avatarStyle}><LockOutlined/></Avatar>
                    <h2>登录</h2>
                </Grid>
                <TextField label='用户名' placeholder='输入用户名' variant="outlined" fullWidth required/>
                <TextField label='密码' placeholder='输入密码' type='password' variant="outlined" fullWidth required/>
                <Button type='submit' color='primary' variant="contained" style={btnstyle} fullWidth>登录</Button>
                <Typography >
                     <Link href="#" >
                        忘记密码 ?
                </Link>
                </Typography>
                <Typography > Do you have an account ?
                     <Link href="#" >
                        注册
                </Link>
                </Typography>
            </Paper>
        </Grid>
    )
}

export default LoginModule