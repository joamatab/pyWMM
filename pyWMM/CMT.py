# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
'''
   Coupled mode theory: setup coupling matrices
   (called by CMTsetup)

   Input:
   umode ....................(mode_array) modes to be coupled
   cpwg ..................... (waveguide) the entire coupler structure

   Output:
   cmtS ..................... (matrix) power coupling matrix
   cmtBK .................... (matrix) coupling coefficients
'''
def CMTmats(umode, cpwg, cmtS, cmtBK):
	n = umode.num;       # number of modes
	#Waveguide wd;        #
	s = np.zeros((n, n))
	k = np.zeros((n, n))
	tmp = np.zeros((n, n))
	p = zeros((n,))
	b = zeros((n,))
	a = np.zeros((n, n))

	#double wl;
	#Polarization pl;
	#Vecform vf;
	#Trifunfield tff;
	#WMM_Mode modei;
	#WMM_Mode modej;

	#Rect r;
    raise ValueError('No modes detected in mode_array') if n <=1

    # Get characteristics of first mode to check other modes
    modei = umode.mode_list[0];
	pl = modei.pol;       # polarization
	wl = modei.wg.lambda; # wavelength
	vf = modei.vform;     # vector form
	if(cpwg.bdmatch(modei.wg) != 0) cmterror("wg.hx, wg.hy");
	p(0) = modei.totpower();
	b(0) = modei.beta;
	for(i=1; i<=n-1; ++i)
	{
		modei = umode(i);
		if(modei.pol != pl) cmterror("pol");
		if(modei.wg.lambda != wl) cmterror("lambda");
		if(pl == VEC)
		{
			if(modei.vform != vf) cmterror("vform");
		}
		if(cpwg.bdmatch(modei.wg) != 0) cmterror("wg.hx, wg.hy");
		p(i) = modei.totpower();
		b(i) = modei.beta;
	}

	for(i=0; i<=n-1; ++i)
	{
		modei = umode(i);
		for(j=0; j<=n-1; ++j)
		{
			if(i==j) tmp(i,j) = p(i);
			else     tmp(i,j) = scalprod(modei, umode(j));
		}
	}
	for(i=0; i<=n-1; ++i)
	{
		for(j=0; j<=n-1; ++j)
		{
			s(i,j) = 0.5*(tmp(i,j)+tmp(j,i))/sqrt(p(i)*p(j));
		}
	}

	for(i=0; i<=n-1; ++i)
	{
		modei = umode(i);
		wd = permdifference(cpwg, modei.wg);
		for(j=0; j<=n-1; ++j)
		{
			modej = umode(j);
			tmp(i,j) = 0.0;
			for(l=0; l<=wd.nx+1; ++l)
			{
				for(m=0; m<=wd.ny+1; ++m)
				{
					if(fabs(wd.n(l,m)) > 1.0e-10)
					{
						r = wd.rectbounds(l,m);
						tmp(i,j) += wd.n(l,m)*(
				twomrecint(modei, EX, modej, EX, r)
			      + twomrecint(modei, EY, modej, EY, r)
			      + twomrecint(modei, EZ, modej, EZ, r));
					}
				}
			}
		}
	}
	for(i=0; i<=n-1; ++i)
	{
		for(j=0; j<=n-1; ++j)
		{
			k(i,j) = 1.0/val_invomep0(wl)
			         /4.0/sqrt(p(i)*p(j))
				 *0.5*(tmp(i,j)+tmp(j,i));
			k(i,j) += s(i,j)
			          *0.5*(b(i) + b(j));
		}
	}

	cmtS  = s
	cmtBK = k
	return cmtS, cmtBK

# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
'''
   Coupled mode theory:
   setup coupling matrices, calculate supermode propagation constants
   and normalized amplitude vectors
'''
def CMTsetup(WMM_ModeArray& umode, // modes to be coupled
	      Waveguide cpwg,       // the entire coupler structure
	      Dmatrix& sigma,       // output: power coupling matrix
	      Dvector& smpc,        // output: supermode propagation constants
	      Dmatrix& smav):       // output: supermode amplitude vectors

# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
'''
   compose CMT supermodes
'''
def CMTsupermodes(WMM_ModeArray& umode,  // modes to be coupled
		   Waveguide cpwg,        // the entire coupler structure
				          // as output by CMTsetup:
	           Dvector smpc,          //   supermode propagation constants
	           Dmatrix smav,          //   supermode amplitude vectors
		   WMM_ModeArray& smode); // output: CMT supermodes

# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
'''
/* Coupled mode theory:
   relative output amplitude of mode o, excitement in mode i */
'''
def CMTamp(int i,              // number of input mode
               int o,              // number of output mode
	       double len,         // device length
	    		           // as output from CMTsetup:
	       Dmatrix& sigma,     //   power coupling matrix
	       Dvector& smpc,      //   supermode propagation constants
	       Dmatrix& smav):     //   supermode amplitude vectors

# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
'''
/* Coupled mode theory:
   relative output power in mode o, excitement in mode i */
'''
def CMTpower(int i,              // number of input mode
                int o,              // number of output mode
		double len,         // device length
				    // as output from CMTsetup:
	        Dmatrix& sigma,     //   power coupling matrix
	        Dvector& smpc,      //   supermode propagation constants
	        Dmatrix& smav):     //   supermode amplitude vectors
# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
'''
/* Coupled mode theory:
   relative output amplitude of mode o,
   two input modes i0, i1, with complex amplitudes c0, c1 */
 '''
def CMTpower2(int i0,            // number of first input mode
	         Complex c0,        //   its amplitude
                 int i1,            // number of second input mode
	         Complex c1,        //   its amplitude
                 int o,             // number of output mode
	         double len,        // device length
	   		            // as output from CMTsetup:
	         Dmatrix& sigma,    //   power coupling matrix
	         Dvector& smpc,     //   supermode propagation constants
	         Dmatrix& smav):    //   supermode amplitude vectors

# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #

'''
/* Coupled mode theory: evaluate field superposition, local intensity
   umode's are assumed to be normalized !!! */
'''
def CMTsz(double x,              // coordinates on the
	     double y,              //   waveguide cross section
	     double z,              // propagation distance
	     WMM_ModeArray& umode,  // modes to be coupled
             int i,                 // number of input mode, power 1 at z=0
	  		            // as output from CMTsetup:
	     Dmatrix& sigma,        //   power coupling matrix
	     Dvector& smpc,         //   supermode propagation constants
	     Dmatrix& smav):        //   supermode amplitude vectors

# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
'''
/* Coupled mode theory: evaluate field superposition, local intensity
   on a mesh in the y-z-plane
   (npy+1)*(npz+1) points between y0<y<y1, 0<z<ldev,
   s(yi, zi) = CMTsz(x, y0+(y1-y0)/npy*yi, z0+ldev/npz*zi),
   umode's are assumed to be normalized !!! */
'''
def CMTprop(double x,              // x-level
	     double y0,             // evaluate between
	     double y1,             //   y0<y<y1,
	     double ldev,           //   0<z<ldev
	     int npy,               // number of
	     int npz,               //   mesh points
	     WMM_ModeArray& umode,  // modes to be coupled
             int i,                 // number of input mode, power 1 at z=0
	                            // as output from CMTsetup:
	     Dmatrix& sigma,        //   power coupling matrix
	     Dvector& smpc,         //   supermode propagation constants
	     Dmatrix& smav,         //   supermode amplitude vectors
	     Dmatrix& s):           // output: power values

# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #

'''
   Coupled mode theory: evaluate field superposition, local intensity
   on a mesh in the y-z-plane
   (npy+1)*(npz+1) points between y0<y<y1, 0<z<ldev,
   s(yi, zi) = CMTsz(x, y0+(y1-y0)/npy*yi, z0+ldev/npz*zi),
   two input modes i0, i1, with complex amplitudes c0, c1,
   umode's are assumed to be normalized !!!
'''

def CMTprop2(double x,              // x-level
	      double y0,             // evaluate between
	      double y1,             //   y0<y<y1,
	      double ldev,           //   0<z<ldev
	      int npy,               // number of
	      int npz,               //   mesh points
	      WMM_ModeArray& umode,  // modes to be coupled
              int i0,                // number of first input mode
	      Complex c0,            //   its amplitude
              int i1,                // number of second input mode
	      Complex c1,            //   its amplitude
	                             // as output from CMTsetup:
	      Dmatrix& sigma,        //   power coupling matrix
	      Dvector& smpc,         //   supermode propagation constants
	      Dmatrix& smav,         //   supermode amplitude vectors
	      Dmatrix& p):           // output: power values
